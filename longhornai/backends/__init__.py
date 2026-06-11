"""Backend implementations, one subpackage/module per execution target.

Per PLAN.md §2.2 the target lifecycle is: cpu -> sim -> rtl -> fpga -> lhsil.
Only ``cpu`` is implemented in M1; the others register the same way when they
land and become selectable through the runtime without changing kernel code.
"""
