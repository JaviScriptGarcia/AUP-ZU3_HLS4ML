# build_accel64_v05.tcl  (v0.5b, 64x64, wider 32->48->96 net)
#
# Synthesizes the BloodMNIST 64x64 AXI-Stream + AXI-Lite wrapper around the
# hls4ml-generated `myproject` core (the wider 32->48->96 net, 93.25% sw),
# producing a Vivado IP catalog entry with ports s_axis_in, s_axis_out,
# s_axi_CTRL. External interface is identical to v0.4 (12288 beats in, 8
# class scores out); only the internal core changed.
#
# Run with:
#   cd 03.hls && vitis_hls -f build_accel64_v05.tcl
#
# Input:  hls4ml C++ project at ../02.hls4ml/output/myproject_wbc64_v05b_prj
# Output: ./ip (note: export_design ignores -output for the packaged IP
#         location and writes to solution1/impl/ip).

set hls4ml_prj "../02.hls4ml/output/myproject_wbc64_v05b_prj"
set firmware   "${hls4ml_prj}/firmware"

open_project -reset myproject_accel64_v05b_prj

set_top myproject_wbc64_v05b_accel

# Core hls4ml-generated files
add_files ${firmware}/myproject.cpp        -cflags "-std=c++14 -I${firmware} -I."
add_files ${firmware}/defines.h
add_files ${firmware}/parameters.h
add_files ${firmware}/nnet_utils
add_files ${firmware}/weights

# Our AXI wrapper (64x64 = 12288 beats)
add_files myproject_wbc64_v05b_accel.cpp    -cflags "-std=c++14 -I${firmware} -I."
add_files myproject_wbc64_v05b_accel.h

open_solution -reset "solution1"

set_part {xczu3eg-sfvc784-2-e}

create_clock -period 10 -name default
config_compile  -name_max_length 80
config_schedule -enable_dsp_full_reg=false

# HLS synthesis
csynth_design

# Export IP (packaged location ends up in solution1/impl/ip regardless)
config_export -format ip_catalog -rtl verilog
export_design  -format ip_catalog -rtl verilog

exit
