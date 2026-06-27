################################################################
# build_bitstream.tcl
#
# Orchestrates the full Vivado flow for v0.4 (BloodMNIST CNN 64x64, wider):
#   1. Create Vivado project targeting the AUP-ZU3 board
#   2. Register the hls4ml-generated IP repository
#   3. Source wbc64_bd.tcl (block design adapted from ICTP)
#   4. Generate the HDL wrapper
#   5. Run synthesis, implementation, bitstream generation
#   6. Export hardware (XSA) with bitstream included
#
# Run from Vivado with:
#   vivado -mode batch -source build_bitstream.tcl
# or from the Vivado Tcl console:
#   cd <this directory>; source build_bitstream.tcl
################################################################

# ---- Configuration ----------------------------------------------------
set script_dir   [file normalize [file dirname [info script]]]
set proj_name    "wbc64_mobilenet"
set proj_dir     [file join $script_dir vivado_prj]
set ip_repo_dir  [file normalize [file join $script_dir .. 03.hls myproject_accel64_prj solution1 impl ip]]
set bd_script    [file join $script_dir wbc64_bd.tcl]
set design_name  "machineLearning_bd"
set part         "xczu3eg-sfvc784-2-e"
set board_part   "realdigital.org:aup-zu3-8gb:part0:1.0"
set xsa_name     "bd_wrapper_wbc64.xsa"

puts "INFO: script_dir   = $script_dir"
puts "INFO: ip_repo_dir  = $ip_repo_dir"
puts "INFO: bd_script    = $bd_script"

# ---- Sanity checks ----------------------------------------------------
if {![file isdirectory $ip_repo_dir]} {
    puts "ERROR: IP repo dir not found: $ip_repo_dir"
    puts "       Run the hls4ml notebook stage (02.hls4ml) first."
    return 1
}
if {![file isfile $bd_script]} {
    puts "ERROR: block-design script not found: $bd_script"
    return 1
}

# ---- Step 1: project ---------------------------------------------------
if {[file isdirectory $proj_dir]} {
    puts "INFO: Removing previous project dir $proj_dir"
    file delete -force $proj_dir
}
create_project $proj_name $proj_dir -part $part
# Try to set the board if its files are installed; not fatal if missing
if {[catch {set_property board_part $board_part [current_project]} err]} {
    puts "WARNING: could not set board_part '$board_part': $err"
    puts "         Synthesis will use the part-only target."
}

# ---- Step 2: register IP repo ------------------------------------------
set_property ip_repo_paths $ip_repo_dir [current_project]
update_ip_catalog
puts "INFO: IP catalog updated with hls4ml IP from $ip_repo_dir"

# ---- Step 3: source the block design ----------------------------------
puts "INFO: Sourcing block design $bd_script"
source $bd_script

# Validate and save BD
save_bd_design
validate_bd_design

# ---- Step 4: HDL wrapper ----------------------------------------------
set bd_file [get_files ${design_name}.bd]

# Force the BD to synthesise as a single global run instead of spawning
# 9 parallel OOC IP runs. With OOC enabled this host (15 GB RAM) OOMs
# because Vivado does not honour -jobs or maxThreads for OOC sub-runs.
# Global mode = one synth_1, peak RAM bounded.
set_property synth_checkpoint_mode None $bd_file
puts "INFO: BD synth_checkpoint_mode set to None (global synthesis, no OOC runs)"

make_wrapper -files $bd_file -top
add_files -norecurse [file join $proj_dir ${proj_name}.gen sources_1 bd $design_name hdl ${design_name}_wrapper.v]
set_property top ${design_name}_wrapper [current_fileset]
update_compile_order -fileset sources_1

# ---- Step 5: synth + impl + bitstream ---------------------------------
#
# v0.4: the host now has 32 GB RAM (v0.3 ran on 15 GB and needed heavy
# throttling). We keep the BD in global synth mode (synth_checkpoint_mode
# None above) so the RAM peak stays bounded and predictable, but we raise
# the internal thread pool and job count to use the extra cores/RAM.
set_param general.maxThreads 8

puts "INFO: Launching synthesis (32 GB host, 8 threads / 4 jobs)"
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} {
    puts "ERROR: synthesis failed. Check logs in $proj_dir/${proj_name}.runs/synth_1/"
    return 1
}

puts "INFO: Launching implementation + bitstream"
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property STATUS [get_runs impl_1]] ne "write_bitstream Complete!"} {
    puts "ERROR: implementation/bitstream failed. Check logs in $proj_dir/${proj_name}.runs/impl_1/"
    return 1
}

# ---- Step 6: export hardware -------------------------------------------
open_run impl_1
set xsa_path [file join $script_dir $xsa_name]
write_hw_platform -fixed -include_bit -force -file $xsa_path
puts "INFO: XSA exported to $xsa_path"

puts ""
puts "================================================================"
puts " Build complete."
puts " XSA: $xsa_path"
puts " Bitstream is embedded inside the XSA."
puts "================================================================"
