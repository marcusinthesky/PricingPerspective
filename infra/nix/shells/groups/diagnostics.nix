# Runtime cost, measured. Both tools here exist only to emit a number about how
# long something took or where the time went — nothing else. Sizing the codebase
# rather than its execution is discovery, and lives in search.nix.
# Descriptor convention: see groups/shell.nix.
{ pkgs }:
{
  description = "Runtime cost: command benchmarking and live Python profiling.";
  packages = with pkgs; [
    hyperfine # `hyperfine`: statistical benchmarking — `--warmup 2 'a' 'b'` to compare, `-L k 1,5,10` to sweep a parameter
    # doCheck: py-spy's `test_thread_names` reads per-thread names from /proc,
    # which the Nix build sandbox restricts, so it panics at build time. The
    # runtime binary is unaffected — do not re-enable.
    (pkgs.py-spy.overrideAttrs (_: {
      doCheck = false;
    })) # `py-spy top|record|dump --pid PID`: sampling profiler, no code change. Needs ptrace; on a hung process run `dump` first
  ];
}
