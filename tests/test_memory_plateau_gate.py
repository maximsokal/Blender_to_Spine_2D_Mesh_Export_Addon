from pathlib import Path
from tools.run_memory_plateau_gate import build_command, evaluate_plateau

def test_memory_command_uses_fresh_background_debug_process(tmp_path):
    command=build_command("blender",tmp_path/"fixture.blend",tmp_path/"outputs",tmp_path/"report.json",warmup=3,iterations=50,sample_every=5)
    assert command[:4]==["blender","--background","--factory-startup","--debug-memory"]; assert "--log-show-memory" in command; assert command[command.index("--iterations")+1]=="50"

def test_plateau_accepts_cache_plateau_and_rejects_linear_leak():
    plateau={"samples":[{"iteration":index,"rss_bytes":100_000_000+delta} for index,delta in enumerate((0,12_000_000,20_000_000,21_000_000,20_500_000,21_200_000))]}
    result=evaluate_plateau(plateau,maximum_tail_growth_bytes=2_000_000,maximum_slope_bytes_per_sample=1_000_000); assert result["compatible"] is True
    leak={"samples":[{"iteration":index,"rss_bytes":100_000_000+index*8_000_000} for index in range(8)]}
    result=evaluate_plateau(leak,maximum_tail_growth_bytes=20_000_000,maximum_slope_bytes_per_sample=2_000_000); assert result["compatible"] is False
