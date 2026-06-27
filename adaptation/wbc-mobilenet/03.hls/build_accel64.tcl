# build_accel64.tcl  (v0.4, 64x64)
#
# Synthesizes the BloodMNIST 64x64 AXI-Stream + AXI-Lite wrapper around the
# hls4ml-generated `myproject` core (the wider 24->32->64 net), producing a
# Vivado IP catalog entry with ports s_axis_in, s_axis_out, s_axi_CTRL.
#
# Run with:
#   cd 03.hls && vitis_hls -f build_accel64.tcl
#
# Input:  hls4ml C++ project at ../02.hls4ml/output/myproject_wbc64_prj
# Output: ./ip64/ with the packaged IP.

set hls4ml_prj "../02.hls4ml/output/myproject_wbc64_prj"
set firmware   "${hls4ml_prj}/firmware"

open_project -reset myproject_accel64_prj

set_top myproject_wbc64_accel

# Core hls4ml-generated files
add_files ${firmware}/myproject.cpp        -cflags "-std=c++14 -I${firmware} -I."
add_files ${firmware}/defines.h
add_files ${firmware}/parameters.h
add_files ${firmware}/nnet_utils
add_files ${firmware}/weights

# Our AXI wrapper (64x64 = 12288 beats)
add_files myproject_wbc64_accel.cpp        -cflags "-std=c++14 -I${firmware} -I."
add_files myproject_wbc64_accel.h

open_solution -reset "solution1"

set_part {xczu3eg-sfvc784-2-e}

create_clock -period 10 -name default
config_compile  -name_max_length 80
config_schedule -enable_dsp_full_reg=false

# HLS synthesis
csynth_design

# Export IP to ./ip64/
config_export -format ip_catalog -rtl verilog -output ./ip64
export_design  -format ip_catalog -rtl verilog -output ./ip64

exit
