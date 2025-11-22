# GPhub-kit-benchmarks

## RUN ANALYSIS

``` shell
cd ~/Documents/GitHub/PhD/GPhub-paper/GPhub-kit
source .venv/bin/activate
uv add --dev gphubkit
cd ~/Documents/GitHub/PhD/GPhub-paper/GPhub-kit/benchmarks/BM/BM_01
gphubkit run bm --id 1
gphubkit postprocess
```
