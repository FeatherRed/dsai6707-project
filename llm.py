"""LLMDE controller: control contract, membership repair, prompts, HTTPS
transport and the consultation policy.

The controller turns statistics-only state (see state.py) into a full
configuration for the shared DE kernel (see de.py): exactly K nonempty groups
covering all stable slot IDs, and one joint (mutation, crossover, F, CR)
configuration per group. Live consultations use a Chat Completions HTTPS
endpoint configured through environment variables.
"""
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import http.client
import json
import math
import os
import random
import re
import socket
import threading
import time
from urllib.parse import urlsplit

from de import CROSSOVERS, MUTATIONS, fallback_groups, integer, validate_groups
from state import validate_state

SCHEMA_VERSION = "llm-de-assignment-v1"


def redact_text(value, secrets=(), limit=16000):
    """Bounded diagnostics without credentials."""
    text = str(value)
    for secret in (*secrets, os.environ.get("GLM_API_KEY", "")):
        if secret:
            text = text.replace(str(secret), "<REDACTED>")
    text = re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <REDACTED>", text)
    text = re.sub(r'''(?i)((?:api[_-]?key|access[_-]?token|authorization)["']?\s*[:=]\s*["']?)[^\s,"'}]+''',
                  r"\1<REDACTED>", text)
    return text[:limit]


class ControlStop(RuntimeError):
    """Expected incomplete run; reason is safe to persist."""


class TransportError(RuntimeError):
    def __init__(self, reason, fatal=False, *, status=None, provider_error=None, retry_after=None):
        super().__init__(redact_text(reason))
        self.fatal = fatal
        self.status = status
        self.provider_error = provider_error
        self.retry_after = retry_after

    def details(self):
        result = {"error": str(self)}
        if self.status is not None:
            result["http_status"] = self.status
        if self.provider_error is not None:
            result["provider_error"] = self.provider_error
        if self.retry_after is not None:
            result["retry_after_seconds"] = self.retry_after
        return result


# GLM business codes requiring an account/key/model/quota change, not a fast retry.
GLM_ACCOUNT_ERRORS = {"1113", "1308", "1309", "1310", "1311", "1313", "1314", "1315",
                      "1316", "1317", "1318", "1319", "1320", "1321"}


def provider_error(raw, key):
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError):
        return {"message": "Non-JSON or truncated error body omitted"}
    error = value.get("error", value) if isinstance(value, dict) else {}
    if not isinstance(error, dict):
        return {"message": "Unrecognized API error body omitted"}
    result = {}
    for field in ("code", "message"):
        if type(error.get(field)) in (str, int, float):
            result[field] = redact_text(error[field], secrets=(key,), limit=1000)
    return result or {"message": "API returned no error code/message"}


def retry_after_seconds(value):
    if not value:
        return None
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            seconds = (when - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


@dataclass
class RequestBudget:
    limit: int
    attempted: int = 0

    def consume(self):
        if self.attempted >= self.limit:
            raise ControlStop("request_budget_exhausted")
        self.attempted += 1


def settings_from_env():
    """Read the three required live-LLM environment variables."""
    result = {"url": os.environ.get("GLM_CHAT_COMPLETIONS_URL", ""),
              "model": os.environ.get("GLM_MODEL", "")}
    if not all(result.values()) or not os.environ.get("GLM_API_KEY"):
        raise ValueError("Live LLMDE requires GLM_CHAT_COMPLETIONS_URL, GLM_MODEL and GLM_API_KEY")
    validate_url(result["url"])
    return result


def validate_url(url):
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("Endpoint must be an HTTPS URL without credentials, query or fragment")
    return parts


def request_body(model, msgs, options):
    body = {"model": model, "messages": msgs, "stream": False,
            **{key: options[key] for key in ("temperature", "top_p", "max_tokens")}}
    if options.get("reasoning_effort") is not None:
        body["reasoning_effort"] = options["reasoning_effort"]
    if options.get("response_format") is not None:
        body["response_format"] = {"type": options["response_format"]}
    return body


class HTTPTransport:
    """Opt-in HTTPS Chat Completions transport with explicit attempt/time budgets."""

    def __init__(self, *, live, settings, connect_timeout, read_timeout, total_timeout):
        if not live:
            raise ValueError("Real transport requires explicit --live")
        self.parts = validate_url(settings["url"])
        self.key = os.environ.get("GLM_API_KEY")
        if not self.key or not settings["model"]:
            raise ValueError("Missing live credentials/model")
        self.connect_timeout, self.read_timeout = connect_timeout, read_timeout
        # Total network wall time across all attempts in THIS run, not optimizer time.
        self.remaining_seconds = total_timeout
        self.network_seconds = 0.0
        self.network_wait_seconds = 0.0

    def wait(self, seconds):
        if seconds > self.remaining_seconds:
            raise TransportError("retry_wait_exceeds_network_time_budget", fatal=True)
        started = time.perf_counter()
        try:
            # Interruptible pacing/backoff; does not count as an HTTP attempt.
            end = started + seconds
            while (remaining := end - time.perf_counter()) > 0:
                time.sleep(min(remaining, 0.25))
        finally:
            elapsed = time.perf_counter() - started
            self.network_wait_seconds += elapsed
            self.network_seconds += elapsed
            self.remaining_seconds -= elapsed

    def send(self, body):
        if self.remaining_seconds <= 0:
            raise TransportError("network_time_budget_exhausted", fatal=True)
        started = time.perf_counter()
        deadline = started + self.remaining_seconds
        connection = http.client.HTTPSConnection(self.parts.hostname, self.parts.port,
                                                 timeout=min(self.connect_timeout, self.remaining_seconds))
        watchdog = None
        try:
            connection.connect()
            active_socket = connection.sock

            def expire():
                # Absolute deadline, including slow/dripping headers or body.
                try:
                    active_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            watchdog = threading.Timer(max(0.001, deadline - time.perf_counter()), expire)
            watchdog.daemon = True
            watchdog.start()

            def set_read_timeout():
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError()
                active_socket.settimeout(min(self.read_timeout, remaining))
            set_read_timeout()
            connection.request("POST", self.parts.path or "/", body=json.dumps(body).encode(),
                               headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
            set_read_timeout()
            response = connection.getresponse()
            status = response.status
            if status >= 400:
                # Keep only bounded, redacted code/message. Never save raw error
                # bodies, arbitrary headers, cookies or request Authorization.
                try:
                    chunks, size = [], 0
                    while size <= 16384:
                        set_read_timeout()
                        chunk = response.read1(min(4096, 16385 - size))
                        if not chunk:
                            break
                        chunks.append(chunk)
                        size += len(chunk)
                    error = provider_error(b"".join(chunks)[:16384], self.key)
                except (OSError, http.client.HTTPException):
                    error = {"message": "Error body unavailable"}
                fatal = status in (401, 403) or (400 <= status < 500 and status not in (408, 429))
                # Use GLM-specific business codes only for its known endpoints.
                if self.parts.hostname in ("open.bigmodel.cn", "api.z.ai"):
                    fatal |= error.get("code") in GLM_ACCOUNT_ERRORS
                raise TransportError(f"http_{status}", fatal=fatal, status=status, provider_error=error,
                                     retry_after=retry_after_seconds(response.getheader("Retry-After")))
            chunks, size = [], 0
            while True:
                set_read_timeout()
                chunk = response.read1(65536)
                if not chunk:
                    break
                size += len(chunk)
                if size > 2_000_000:
                    raise TransportError("response_too_large")
                chunks.append(chunk)
            if time.perf_counter() >= deadline:
                raise TransportError("network_time_budget_exhausted", fatal=True)
            try:
                result = json.loads(b"".join(chunks))
                choice = result["choices"][0]
                content = choice["message"]["content"]
                # Some reasoning models return null after consuming the token
                # cap. Preserve usage/finish_reason; the control validator will
                # still reject the empty decision rather than invent a config.
                if content is None:
                    content = ""
                if not isinstance(content, str):
                    raise ValueError()
                return {"content": content, "usage": result.get("usage") or "unknown",
                        "finish_reason": choice.get("finish_reason"), "response_model": result.get("model")}
            except (ValueError, KeyError, TypeError, IndexError):
                raise TransportError("invalid_api_envelope") from None
        except TransportError:
            raise
        except (TimeoutError, socket.timeout):
            if time.perf_counter() >= deadline:
                raise TransportError("network_time_budget_exhausted", fatal=True) from None
            raise TransportError("network_timeout") from None
        except (OSError, http.client.HTTPException):
            if time.perf_counter() >= deadline:
                raise TransportError("network_time_budget_exhausted", fatal=True) from None
            raise TransportError("network_error") from None
        finally:
            if watchdog is not None:
                watchdog.cancel()
            connection.close()
            elapsed = time.perf_counter() - started
            self.network_seconds += elapsed
            self.remaining_seconds -= elapsed


def _read_payload(content):
    if not isinstance(content, str):
        raise ValueError("Response content must be text")
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) < 3 or lines[0].strip().lower() not in ("```", "```json") or lines[-1].strip() != "```":
            raise ValueError("Malformed Markdown fence")
        content = "\n".join(lines[1:-1])

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(content, object_pairs_hook=unique_keys)


def validate_assignment(payload, n, k=3):
    """Decode wire labels (assignment + configs) to the groups execution contract."""
    if not isinstance(payload, dict) or set(payload) != {"assignment", "configs"}:
        raise ValueError("Expected exactly assignment and configs")
    labels, configs = payload["assignment"], payload["configs"]
    if not isinstance(labels, list) or len(labels) != n:
        raise ValueError(f"assignment must contain exactly {n} group labels in slot ID order")
    for label in labels:
        integer(label, "group label", 0, k - 1)
    if set(labels) != set(range(k)):
        raise ValueError("All K groups must be nonempty")
    if not isinstance(configs, list) or len(configs) != k:
        raise ValueError(f"configs must contain exactly {k} configurations")
    groups = []
    for group_id, config in enumerate(configs):
        if not isinstance(config, dict) or set(config) != {"mutation", "crossover", "F", "CR"}:
            raise ValueError("Unexpected or missing configuration fields")
        groups.append({"members": [i for i, label in enumerate(labels) if label == group_id], **config})
    return validate_groups({"groups": groups}, n, k)


REPAIR_VERSION = "random-membership-repair-v1"


def parse_or_repair(content, n, k, rng):
    """User-authorized, auditable repair of membership only; parameters stay strict."""
    payload = _read_payload(content)
    try:
        return validate_assignment(payload, n, k), None
    except ValueError as exc:
        original_error = str(exc)
    if not isinstance(payload, dict) or set(payload) != {"assignment", "configs"}:
        raise ValueError("Membership repair requires assignment and configs")
    labels = payload["assignment"]
    if not isinstance(labels, list) or n < k:
        raise ValueError("Membership repair requires an assignment array and N>=K")
    # Validate all model-selected operators/parameters before any random draws.
    validate_assignment({"assignment": [i % k for i in range(n)], "configs": payload["configs"]}, n, k)
    repaired = [label if type(label) is int and 0 <= label < k else None for label in labels[:n]]
    repaired.extend([None] * (n - len(repaired)))
    available = sorted({label for label in repaired if label is not None}) or list(range(k))
    changes = []
    for i, label in enumerate(repaired):
        if label is None:
            repaired[i] = rng.choice(available)
            changes.append({"id": i, "reason": "missing_label" if i >= len(labels) else "invalid_label",
                            "to_group": repaired[i]})
    # Keep all K groups nonempty. Move only from groups that can spare a member.
    for empty in range(k):
        if empty not in repaired:
            counts = [repaired.count(g) for g in range(k)]
            eligible = [i for i, label in enumerate(repaired) if counts[label] > 1]
            chosen = rng.choice(eligible)
            previous = repaired[chosen]
            repaired[chosen] = empty
            changes.append({"id": chosen, "reason": "fill_empty_group", "from_group": previous, "to_group": empty})
    result = validate_assignment({"assignment": repaired, "configs": payload["configs"]}, n, k)
    report = {"version": REPAIR_VERSION, "original_error": original_error,
              "original_assignment_length": len(labels), "expected_length": n,
              "discarded_suffix_labels": max(0, len(labels) - n),
              "changes": changes, "changed_ids": sorted({change["id"] for change in changes}),
              "repaired_assignment": repaired}
    return result, report


PROMPT_VERSION = "joint-statistics-v1"
SYSTEM = """Control DE for an unknown continuous minimization problem using observed statistics.
Form exactly K nonempty groups of stable slot IDs and choose one joint DE configuration per group.
Use population progress/diversity and individual quality, improvement, stagnation and distance statistics together.
Similar fitness does not imply similar search needs. Groups have no predefined rank tiers, sizes or roles.
Only statistics are provided: no raw decision vectors, function identity, known optimum or landscape parameters.
All fitness gaps and gains use a fixed initial fitness IQR scale and log(1+ratio); positive gain is better.
Population columns describe budget used, cumulative and last-window best gain, mean gap, fitness spread,
RMS diversity, diversity trend (-1 contraction to +1 expansion), recent success fraction, best stagnation
in H-generation units (capped at 5), fraction stalled at least H generations, mean absolute adjacent-coordinate
sample correlation, and max dimension variance / total variance. Sample correlations are not objective interactions.
Individual rank is the fraction strictly better (ties equal); rank and gap smaller is better.
Individual gain/success refer to the observed window. stall_H is individual stagnation/H capped at 5.
d_best and d_center are RMS-normalized distances to best and centroid; d_knn averages 3 nearest neighbors.
Smaller d_knn means more crowded. prev_group refers only to the previous decision window; -1 means none.
Null means unobserved, not zero performance. Missing credit means no retained evidence, not failure.
last_groups reports the previous group's configuration, evaluated count, success fraction and log mean gain.
credit covers the 4 most recently used full configurations over 50 generations. Counts measure evidence strength.
Credit is observational: different groups/stages/settings are not controlled comparisons of individual operators.
Retain productive configurations; consider regrouping or adjusting stalled groups while balancing exploration.
All donors come from the entire parent population. Random indices are distinct and exclude the target;
best-based methods also exclude best from random donors. Greedy selection is synchronous; IDs persist.
rand/1: xr1+F*(xr2-xr3); rand/2 adds F*(xr4-xr5).
best/1: xbest+F*(xr1-xr2); best/2 adds F*(xr3-xr4).
current-to-best/1: xi+F*(xbest-xi)+F*(xr1-xr2).
Out-of-bounds mutant dimensions are uniformly resampled in normalized [0,1].
Binomial copies independent dimensions with probability CR and forces one dimension.
Exponential copies a circular contiguous segment from a random start, at least one dimension, extending with CR.
F must be finite, 0<F<=2; CR finite, 0<=CR<=1. Membership/configurations persist H generations or until budget ends.
If assigned_crossovers is non-null, configs[j].crossover must equal its j-th entry.
Return ONLY a JSON object with exactly two keys: assignment and configs.
assignment is a list of exactly N integers in [0,K-1]; element i is the group for slot ID i; all K labels must occur.
configs is a list of exactly K objects, each with exactly mutation, crossover, F, CR. Entry g configures group g.
Use only the supplied mutation/crossover pools. Check assignment length and nonempty groups before returning.
No explanations or extra fields."""


def messages(state, k, interval):
    validate_state(state)
    task = {"prompt_version": PROMPT_VERSION, "schema_version": SCHEMA_VERSION,
            "K": k, "H": interval, "remaining_evaluations": max(0, state["max_fes"] - state["fes"]),
            "mutation_pool": list(MUTATIONS), "crossover_pool": list(CROSSOVERS),
            "assigned_crossovers": None, "state": state}
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(task, ensure_ascii=True, allow_nan=False, separators=(",", ":"))}]


def prompt_hash(msgs):
    return hashlib.sha256(json.dumps(msgs, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class Controller:
    """Consultation policy for LLMDE.

    kind="mock" exercises the full pipeline offline with random legal
    configurations -- explicitly NOT a quality model. kind="llm" queries a
    Chat Completions endpoint for the joint (grouping, mutation, crossover,
    F, CR) decision, with bounded attempts, retries and fallback to the last
    valid configuration.
    """

    def __init__(self, kind, *, seed, k=3, interval=10, transport=None, budget=None,
                 settings=None, options=None, retries=0):
        if kind not in ("mock", "llm"):
            raise ValueError("Controller kind must be 'mock' or 'llm'")
        self.kind, self.k, self.interval = kind, k, interval
        self.mock_rng = random.Random(seed + 3)
        self.retry_rng = random.Random(seed + 4)
        self.membership_rng = random.Random(seed + 8001)
        self.transport, self.budget = transport, budget
        self.settings, self.options = settings or {}, options or {}
        self.retries = retries
        self.previous = None
        self.last_attempt_start = None
        self.consultations = 0
        self.attempted_requests = 0
        self.successful_requests = 0
        self.valid_configurations = 0
        self.repaired_consultations = 0
        self.repair_changed_individuals_total = 0
        self.repair_discarded_suffix_labels_total = 0
        self.parse_failures = 0
        self.http_failures = 0
        self.fallbacks = 0
        self.known_total_tokens = 0
        self.rate_limit_responses = 0
        self.output_token_limit_responses = 0
        self.last_transport_error = None

    def _log(self, **fields):
        if self.options.get("verbose_control", False):
            print(json.dumps({"llm_control": fields}, ensure_ascii=False), flush=True)

    def _wait(self, seconds):
        if hasattr(self.transport, "wait"):
            self.transport.wait(seconds)
        else:
            time.sleep(seconds)

    def _mock(self, state):
        ids = list(range(state["N"]))
        self.mock_rng.shuffle(ids)
        groups = []
        for i in range(self.k):
            groups.append({"members": ids[i::self.k], "mutation": self.mock_rng.choice(MUTATIONS),
                           "crossover": self.mock_rng.choice(CROSSOVERS),
                           "F": self.mock_rng.uniform(0.3, 0.9), "CR": self.mock_rng.uniform(0.1, 0.9)})
        return validate_groups({"groups": groups}, state["N"], self.k)

    def consult(self, state):
        self.consultations += 1
        if self.kind == "mock":
            payload = self._mock(state)
            self.previous = payload
            return payload
        msgs = messages(state, self.k, self.interval)
        body = request_body(self.settings["model"], msgs, self.options)
        self._log(consultation=self.consultations, fes=state["fes"], prompt_hash=prompt_hash(msgs))
        last_transport_error, next_delay = None, 0.0
        for retry_index in range(self.retries + 1):
            if getattr(self.transport, "remaining_seconds", 1) <= 0:
                raise ControlStop("network_time_budget_exhausted")
            if self.budget.attempted >= self.budget.limit:
                raise ControlStop("request_budget_exhausted")
            gap = self.options.get("request_interval_seconds", 0.0)
            pace = 0.0 if self.last_attempt_start is None else max(0.0, gap - (time.perf_counter() - self.last_attempt_start))
            delay = max(pace, next_delay)
            if delay > 0:
                self._log(consultation=self.consultations, fes=state["fes"], waiting_seconds=delay,
                          reason="retry_backoff" if next_delay else "request_pacing")
                try:
                    self._wait(delay)
                except TransportError as exc:
                    raise ControlStop(str(exc)) from None
                except KeyboardInterrupt:
                    raise
            self.budget.consume()
            self.attempted_requests += 1
            attempt_number = self.budget.attempted
            started = time.perf_counter()
            self.last_attempt_start = started
            self._log(consultation=self.consultations, fes=state["fes"], attempt=attempt_number, status="requesting")
            try:
                response = self.transport.send(body)
                self.successful_requests += 1
                usage = response.get("usage")
                if isinstance(usage, dict) and type(usage.get("total_tokens")) is int and usage["total_tokens"] >= 0:
                    self.known_total_tokens += usage["total_tokens"]
                if response.get("finish_reason") == "length":
                    self.output_token_limit_responses += 1
                payload, repair = parse_or_repair(response["content"], state["N"], self.k, self.membership_rng)
                if repair is not None:
                    self.repaired_consultations += 1
                    self.repair_changed_individuals_total += len(repair["changed_ids"])
                    self.repair_discarded_suffix_labels_total += repair["discarded_suffix_labels"]
                    self._log(consultation=self.consultations, fes=state["fes"], status="membership_repaired",
                              changed_ids=repair["changed_ids"], discarded_suffix_labels=repair["discarded_suffix_labels"])
                self.valid_configurations += 1
                self.previous = payload
                self._log(consultation=self.consultations, fes=state["fes"], attempt=attempt_number,
                          status="valid_configuration", wall_seconds=round(time.perf_counter() - started, 3),
                          finish_reason=response.get("finish_reason"))
                return payload
            except TransportError as exc:
                self.http_failures += 1
                if exc.status == 429:
                    self.rate_limit_responses += 1
                self.last_transport_error = exc.details()
                self._log(consultation=self.consultations, fes=state["fes"], **exc.details())
                if exc.fatal:
                    raise ControlStop(str(exc)) from None
                last_transport_error = exc
            except (ValueError, TypeError, KeyError) as exc:
                last_transport_error = None
                parse_error = redact_text(exc, limit=500)
                self.parse_failures += 1
                self._log(consultation=self.consultations, fes=state["fes"], status="invalid_configuration",
                          parse_error=parse_error)
                # Membership repair was attempted above. Unparseable JSON or
                # illegal operator/parameter choices still require model correction.
                # Retry carries the bounded validation error only. Never echo
                # arbitrary malformed output into the statistics-only prompt.
                body = {**body, "messages": [*msgs, {"role": "user", "content": (
                            "Your previous response failed validation: " + parse_error
                            + ". Return the complete corrected JSON: assignment must have exactly N integer labels "
                              "in slot ID order, every group nonempty, and configs must have exactly K legal entries. "
                              "Keep all configuration constraints. JSON only.")}]}
            except KeyboardInterrupt:
                raise
            base = self.options.get("retry_backoff_seconds", 2.0)
            maximum = self.options.get("retry_backoff_max_seconds", 60.0)
            next_delay = min(maximum, base * (2 ** retry_index) * self.retry_rng.uniform(1.0, 1.25))
            if last_transport_error is not None and last_transport_error.retry_after is not None:
                # Never shorten the provider's Retry-After to our jitter cap.
                next_delay = max(next_delay, last_transport_error.retry_after)
        if self.budget.attempted >= self.budget.limit:
            raise ControlStop("request_budget_exhausted")
        if last_transport_error is not None:
            raise ControlStop(str(last_transport_error)) from None
        if self.options.get("fail_on_fallback", False):
            raise ControlStop("invalid_configuration_after_retries")
        # A failed consultation uses exactly the previous legal configuration.
        payload = copy.deepcopy(self.previous or fallback_groups(state["N"], self.k))
        self.fallbacks += 1
        self._log(consultation=self.consultations, fes=state["fes"], status="fallback",
                  source="previous_valid" if self.previous else "deterministic_id_cohorts")
        return payload

    def statistics(self):
        return {"consultations": self.consultations,
                "attempted_requests": self.attempted_requests,
                "successful_requests": self.successful_requests,
                "valid_configurations": self.valid_configurations,
                "membership_repair_version": REPAIR_VERSION,
                "repaired_consultations": self.repaired_consultations,
                "repair_changed_individuals_total": self.repair_changed_individuals_total,
                "repair_discarded_suffix_labels_total": self.repair_discarded_suffix_labels_total,
                "parse_failures": self.parse_failures,
                "http_failures": self.http_failures,
                "fallbacks": self.fallbacks,
                "known_total_tokens": self.known_total_tokens,
                "rate_limit_responses": self.rate_limit_responses,
                "output_token_limit_responses": self.output_token_limit_responses,
                "last_transport_error": self.last_transport_error}
