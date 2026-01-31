class DiagnosticTools:
    def run(self, raw_tx: dict) -> str:
        findings = []

        amount = raw_tx.get("amount", 0)
        hour = raw_tx.get("hour_of_day", -1)
        tx_1h = raw_tx.get("tx_count_1h", 0)

        if amount > 100_000:
            findings.append("High transaction amount")

        if hour in [0, 1, 2, 3, 4]:
            findings.append("Transaction during off-hours")

        if tx_1h > 10:
            findings.append("High transaction frequency in last hour")

        if not findings:
            findings.append("No deterministic red flags")

        return "; ".join(findings)
