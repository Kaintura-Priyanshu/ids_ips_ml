"""
Intrusion Prevention System (IPS) decision + action engine.

Given a detection result (predicted class, confidence, anomaly score, source
IP), decides ALLOW / ALERT / RATE_LIMIT / BLOCK per the policy thresholds in
config.yaml, tracks a blocklist with expiry + repeat-offense escalation, and
logs every action. Blocking is simulated (logged + printed as the firewall
rule that *would* be issued) rather than touching real system firewall
rules — see README for why.
"""
import json
import time
from pathlib import Path
from datetime import datetime, timedelta


class IPSEngine:
    def __init__(self, policy_cfg: dict, logging_cfg: dict):
        self.policy = policy_cfg
        self.log_dir = Path(logging_cfg["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.blocklist_path = Path(logging_cfg["blocklist_file"])
        self.action_log_path = Path(logging_cfg["action_log_file"])
        self.blocklist = self._load_blocklist()
        self.offense_counts = {}

    # ---------- persistence ----------
    def _load_blocklist(self):
        if self.blocklist_path.exists():
            return json.loads(self.blocklist_path.read_text())
        return {}

    def _save_blocklist(self):
        self.blocklist_path.write_text(json.dumps(self.blocklist, indent=2, default=str))

    def _log_action(self, line: str):
        ts = datetime.utcnow().isoformat()
        with open(self.action_log_path, "a") as f:
            f.write(f"[{ts}] {line}\n")

    # ---------- decision ----------
    def is_blocked(self, src_ip: str) -> bool:
        entry = self.blocklist.get(src_ip)
        if not entry:
            return False
        if entry.get("permanent"):
            return True
        expires = datetime.fromisoformat(entry["expires_at"])
        if datetime.utcnow() > expires:
            del self.blocklist[src_ip]
            self._save_blocklist()
            return False
        return True

    def decide(self, src_ip: str, predicted_class: str, confidence: float, anomaly_score: float):
        """Returns (action, reason) where action in {ALLOW, ALERT, RATE_LIMIT, BLOCK}."""
        if self.is_blocked(src_ip):
            return "BLOCK", f"{src_ip} already on active blocklist"

        is_anomalous = anomaly_score < self.policy["anomaly_threshold"]

        if predicted_class == "BENIGN" and not is_anomalous:
            return "ALLOW", "benign traffic, no anomaly flag"

        if predicted_class == "BENIGN" and is_anomalous:
            return "ALERT", f"anomaly score {anomaly_score:.4f} below threshold, no known attack class"

        # known attack class predicted
        if confidence >= self.policy["block_threshold"]:
            return "BLOCK", f"predicted={predicted_class} confidence={confidence:.3f} >= block_threshold"
        if confidence >= self.policy["rate_limit_threshold"]:
            return "RATE_LIMIT", f"predicted={predicted_class} confidence={confidence:.3f} >= rate_limit_threshold"
        return "ALERT", f"predicted={predicted_class} confidence={confidence:.3f} below action thresholds"

    # ---------- enforcement ----------
    def act(self, src_ip: str, action: str, reason: str, predicted_class: str):
        if action == "BLOCK":
            self._block(src_ip, predicted_class, reason)
        elif action == "RATE_LIMIT":
            self._log_action(f"RATE_LIMIT {src_ip} | class={predicted_class} | {reason}")
        elif action == "ALERT":
            self._log_action(f"ALERT {src_ip} | class={predicted_class} | {reason}")
        else:
            pass  # ALLOW -> no log needed to keep noise down

    def _block(self, src_ip: str, predicted_class: str, reason: str):
        self.offense_counts[src_ip] = self.offense_counts.get(src_ip, 0) + 1
        permanent = self.offense_counts[src_ip] >= self.policy["offense_escalation_count"]

        expires_at = datetime.utcnow() + timedelta(minutes=self.policy["block_duration_minutes"])
        self.blocklist[src_ip] = {
            "blocked_at": datetime.utcnow().isoformat(),
            "expires_at": expires_at.isoformat(),
            "permanent": permanent,
            "reason": reason,
            "offense_count": self.offense_counts[src_ip],
        }
        self._save_blocklist()

        rule = f"iptables -A INPUT -s {src_ip} -j DROP"
        status = "PERMANENT" if permanent else f"TEMPORARY ({self.policy['block_duration_minutes']}m)"
        self._log_action(
            f"BLOCK {src_ip} | class={predicted_class} | {reason} | {status} | "
            f"offense#{self.offense_counts[src_ip]} | would run: {rule}"
        )
        if self.policy.get("enforce_real_firewall_rules"):
            self._apply_real_firewall_rule(src_ip)

    def _apply_real_firewall_rule(self, src_ip: str):
        # Intentionally not implemented: wiring this to a real iptables/
        # nftables/cloud security-group call is a deployment-specific
        # decision that shouldn't happen inside a sandboxed demo.
        raise NotImplementedError(
            "Real firewall enforcement is not implemented in this project. "
            "Implement this method for your specific deployment target."
        )
