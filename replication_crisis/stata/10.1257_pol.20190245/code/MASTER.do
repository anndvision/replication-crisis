/******************************************************
CUMULATIVE IMPACTS OF CONDITIONAL CASH TRANSFER PROGRAMS:
EXPERIMENTAL EVIDENCE FROM INDONESIA

MASTER .DO FILE

***CONTENTS***
*1. Setup
*2. Coding datasets for analysis
*3. Calculating RW p-values for main outcomes 
*4. Analysis
******************************************************/

*****************
****1. Setup ****
*****************
clear all 
set more off
pause on 
set matsize 10000
cap log close 

//Enter path where main file folder is located here:
global PKH = ""

//install necessary user-written commands 
foreach package in carryforward estout outreg frmttable {
     capture which `package'
	 if _rc==111 ssc install `package', replace 
}

//ado folder setup 
adopath + "$PKH/ado/"

/**
NOTE: CODE FOR CALCULATING ROMANO-WOLF P-VALUES TAKES A SUBSTANTIAL AMOUNT OF TIME 
TO RUN (UP TO 24 HOURS ON SOME COMPUTERS). 

SET THE FOLLOWING LOCAL EQUAL TO 0 IF YOU DO NOT WANT TO RUN THE .DO FILES
THAT CALCULATE RW P-VALUES (RWOLF_TABLE2-RWOLF_TABLE7).
SET THE LOCAL EQUAL TO 1 IF YOU WOULD LIKE TO RUN THESE FILES.
**/
local do_rwolf = 0


//open log file 
log using "$PKH/logs/pkh_master_log", replace 


***************************************
****2. Coding datasets for analysis****
***************************************
cd "$PKH/code/coding"

//coding village-level outcomes
do "$PKH/code/coding/code_master_village_level_data.do"
	//in: various raw survey datasets 
	//out: village_outcomes_master.dta

//coding household-level outcomes 
do "$PKH/code/coding/code_master_household_data.do"
	//in: various raw survey datasets; village_outcomes_master.dta
	//out: household_allwaves_master.dta

//coding outcomes for married women age 16-49
do "$PKH/code/coding/code_master_woman16to49.do"
	//in: various raw survey datasets; household_allwaves_master.dta
	//out: marriedwoman16to49_allwaves_master.dta

//coding outcomes for children 0-36 months
do "$PKH/code/coding/code_master_child0to36mo.do"
	//in: various raw survey datasets; household_allwaves_master.dta
	//out: child0to36months_allwaves_master.dta, plus output from 
	//		WHO anthropometric module 

//coding outcomes for children 6-15 years 
do "$PKH/code/coding/code_master_child6to15yr.do"
	//in: various raw survey datasets; household_allwaves_master.dta
	//out: child6to15_allwaves_master.dta

//coding attrtition of children 6-15 years 
do "$PKH/code/coding/code_child6to15_survey_attrition.do"
	//in: various raw survey datasets; household_allwaves_master.dta 
	//out: tracking_child6to15_attrition.dta

//coding fertility and child marriage outcomes 
do "$PKH/code/coding/code_child_fertility_tracking.do"
do "$PKH/code/coding/code_child_fertility_marriage_outcomes.do"
	//in: various raw survey datasets; household_allwaves_master.dta
	//out: tracking_child6to15_attrition_for_fertility.dta; 
	//		child6to15_fertility_marriage_outcomes.dta

//coding maternal fertility timing outcomes 
//NOTE: .do file is available, but because of PII contained in raw data, we have only published the de-identified output: mothers_fertility_timing.dta
*do "$PKH/code/coding/code_fertility_timing_outcomes.do"
	//in: various raw survey datasets 
	//out: mothers_fertility_timing.dta

//coding dataset for infant mortality outcomes 
do "$PKH/code/coding/code_master_infant_mortality.do"
	//in: various raw survey datasets 
	//out: infantmortality_allwaves_master.dta 


*************************************************************
****3. Calculating Romano-Wolf p-values for main outcomes****
*************************************************************
//Note: only runs if toggled on above
if `do_rwolf' == 1 {
	do "$PKH/code/analysis/rwolf_table2.do"
	//in: marriedwoman16to49_allwaves_master.dta; child0to36months_allwaves_master.dta
	//out: rwolf_pvalues_table2.xls

	do "$PKH/code/analysis/rwolf_table3.do"
	//in: child6to15_allwaves_master.dta
	//out: rwolf_pvalues_table3.xls

	do "$PKH/code/analysis/rwolf_table4.do"
	//in: child0to36months_allwaves_master.dta
	//out: rwolf_pvalues_table4.xls

	do "$PKH/code/analysis/rwolf_table5.do"
	//in: child6to15_allwaves_master
	//out: rwolf_pvalues_table5.xls

	do "$PKH/code/analysis/rwolf_table6.do"
	//in: child6to15_allwaves_master.dta; child6to15_fertility_marriage_outcomes.dta
	//out: rwolf_pvalues_table6.xls

	do "$PKH/code/analysis/rwolf_table7.do"
	//in: household_allwaves_master.dta
	//out: rwolf_pvalues_table7.xls
}


*************************************************************
****4. Analysis**********************************************
*************************************************************
cd "$PKH/code/analysis"

do "table1.do"
	//in: household_allwaves_master.dta
	//out: table1.tex

do "$PKH/code/analysis/table2.do"
	//in: marriedwoman16to49_allwaves_master.dta; child0to36months_allwaves_master.dta
	//out: table2.tex

do "$PKH/code/analysis/table3.do"
	//in: child6to15_allwaves_master.dta
	//out: table3.tex

do "$PKH/code/analysis/table4.do"
	//in: child0to36months_allwaves_master.dta
	//out: table4.tex

do "$PKH/code/analysis/table5.do"
	//in: child6to15_allwaves_master.dta
	//out: table5.tex

do "$PKH/code/analysis/table6.do"
	//in: child6to15_allwaves_master.dta; child6to15_fertility_marriage_outcomes.dta
	//out: table6.tex

do "$PKH/code/analysis/table7.do"
	//in: household_allwaves_master.dta
	//out: table7.tex


****APPENDIX****
do "$PKH/code/analysis/appendix_figure2.do"
	//in: household_allwaves_master.dta 
	//out: appendix_figure2.tex

do "$PKH/code/analysis/appendix_figure4.do"
	//in: pkhben2013_use.dta
	//out: appendix_figure4.tex

do "$PKH/code/analysis/appendix_figure5.do"
	//in: child0to36months_allwaves_master
	//out: appendix_figure5_stunting.png; appendix_figure5_severe_stunting.png

do "$PKH/code/analysis/appendix_table1.do"
	//in: household_allwaves_master.dta
	//out: appendix_table1.tex

do "$PKH/code/analysis/appendix_table2.do"
	//in: marriedwoman16to49_allwaves_master.dta; infantmortality_allwaves_master.dta; 
	// 		mothers_fertility_timing.dta
	//out: appendix_table2.tex

do "$PKH/code/analysis/appendix_table3.do"
	//in: tracking_child6to15_attrition.dta
	//out: appendix_table3.tex

do "$PKH/code/analysis/appendix_table4.do"
	//in: tracking_child6to15_attrition.dta
	//out: appendix_table4.tex

do "$PKH/code/analysis/appendix_table5ab.do"
	//in: tracking_child6to15_attrition.dta
	//out: appendix_table5a.tex; appendix_table5b.tex

do "$PKH/code/analysis/appendix_table6.do"
	//in: marriedwoman16to49_allwaves_master; child0to36months_allwaves_master.dta; 
	//		child6to15_allwaves_master.dta
	//out: appendix_table6.tex

do "$PKH/code/analysis/appendix_table7.do"
	//in: village_outcomes_master.dta 
	//out: appendix_table7.tex

do "$PKH/code/analysis/appendix_table8.do"
	//in: marriedwoman16to49_allwaves_master.dta; child0to36months_allwaves_master.dta 
	//out: appendix_table8.tex

do "$PKH/code/analysis/appendix_table9.do"
	//in: child6to15_allwaves_master.dta 
	//out: appendix_table9.tex

do "$PKH/code/analysis/appendix_table10.do"
	//in: marriedwoman16to49_allwaves_master.dta; child0to36months_allwaves_master.dta
	//out: appendix_table10.tex

do "$PKH/code/analysis/appendix_table11.do"
	//in: marriedwoman16to49_allwaves_master.dta; child0to36months_allwaves_master.dta
	//out: appendix_table11.tex

do "$PKH/code/analysis/appendix_table12.do"
	//in: child6to15_allwaves_master.dta 
	//out: appendix_table12.tex

do "$PKH/code/analysis/appendix_table13.do"
	//in: child6to15_allwaves_master.dta
	//out: appendix_table13.tex

do "$PKH/code/analysis/appendix_table14.do"
	//in: child6to15_allwaves_master.dta
	//out: appendix_table14.tex

do "$PKH/code/analysis/appendix_table15.do"
	//in: child0to36months_allwaves_master.dta
	//out: appendix_table15.tex

do "$PKH/code/analysis/appendix_table16.do"
	//in: child0to36months_allwaves_master.dta
	//out: appendix_table16.tex

do "$PKH/code/analysis/appendix_table17.do"
	//in: child0to36months_allwaves_master.dta 
	//out: appendix_table17.tex

do "$PKH/code/analysis/appendix_table18.do"
	//in: marriedwoman16to49_allwaves_master.dta; child0to36months_allwaves_master.dta; 
	//	  household_allwaves_master.dta
	//out: appendix_table18.tex

do "$PKH/code/analysis/appendix_table19.do"
	//in: child0to36months_allwaves_master.dta 
	//out: appendix_table19.tex

do "$PKH/code/analysis/appendix_table20.do"
	//in: child0to36months_allwaves_master.dta
	//out: appendix_table20.tex

do "$PKH/code/analysis/appendix_table21.do"
	//in: child6to15_allwaves_master.dta
	//out: appendix_table21.tex

do "$PKH/code/analysis/appendix_table22.do"
	//in: child6to15_allwaves_master
	//out: appendix_table22.tex

do "$PKH/code/analysis/appendix_table23.do"
	//in: child6to15_allwaves_master.dta
	//out: appendix_table23.tex

do "$PKH/code/analysis/appendix_table24.do"
	//in: child6to15_allwaves_master.dta
	//out: appendix_table24.tex

do "$PKH/code/analysis/appendix_table25.do"
	//in: child6to15_allwaves_master.dta
	//out: appendix_table25.tex

do "$PKH/code/analysis/appendix_table26.do"
	//in: child6to15_allwaves_master.dta; child6to15_fertility_marriage_outcomes.dta
	//out: appendix_table26.tex

do "$PKH/code/analysis/appendix_table27.do"
	//in: child6to15_allwaves_master.dta
	//out: appendix_table27.tex

do "$PKH/code/analysis/appendix_table28.do"
	//in: child6to15_fertility_marriage_outcomes.dta
	//out: appendix_table28.tex

do "$PKH/code/analysis/appendix_table29.do"
	//in: household_allwaves_master.dta
	//out: appendix_table29.tex


