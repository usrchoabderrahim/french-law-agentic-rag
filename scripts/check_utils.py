"""Tiny helper: print PASS/FAIL for each check and a final summary."""
RESULTS: list[tuple[str, bool]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))


def show(results, limit: int = 3) -> None:
    for r in results[:limit]:
        print(f"      {r.score:.4f} | p.{r.page_start} | {str(r.section)[:30]:30s} | {r.text[:60]!r}")


def summary() -> None:
    passed = sum(ok for _, ok in RESULTS)
    print(f"\n===== {passed}/{len(RESULTS)} checks passed =====")
    for name, ok in RESULTS:
        if not ok:
            print("  FAILED:", name)
