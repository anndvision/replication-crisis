*** 1. INTRODUCTION ***
/*
Description: 	Generates .tex file for Appendix Table 1 (Household Survey Attrition)
Uses: 			household_allwaves_master.dta
Creates: 		appendix_table1.tex
*/

*** 2. SET ENVIRONMENT ***

version 14.1 
set more off		
clear all			
pause on 			

//Prompt for directory if not filled in above
if "$PKH" == "" {
	display "Enter the directory where the PKH files can be located: " _request(PKH)
	* add a \ if they forgot one
	quietly if substr("$PKH",length("PKH"),1) != "/" & substr("$PKH",length("$PKH"),1) != "\" {
		global PKH = "$PKH\"	
	}
}

local PKH = "$PKH"
local pkhdata = "`PKH'data/raw/"
local output = "`PKH'output"
local latex = "`output'/latex"
local TARGET2 = "$TARGET2"


*** 3. PREPARE DATA & STORE TABLE ***
cd "`PKH'data/coded"
use "household_allwaves_master.dta", clear

//Label values of survey_round for table 
label define round 0 "Baseline" 1 "2-Year" 2 "6-Year"
label values survey_round round

//tabulate and store
estpost tab survey_round L07 if baseline_match == 1 & split_indicator == 0

//extract matrix 
matrix frac = e(b)

//convert this matrix into percentages of baseline count
local denom1 = frac[1,1]
local denom2 = frac[1,5]
local denom3 = frac[1,9]

forvalues i = 1/3 {
	local j = `i' + 4
	local k = `i' + 8
	matrix frac[1,`i'] = (frac[1,`i'] / `denom1')*100
	matrix frac[1,`j'] = (frac[1,`j'] / `denom2')*100
	matrix frac[1,`k'] = (frac[1,`k'] / `denom3')*100
}

estadd matrix frac
eststo A


*** 4. OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;

esttab A using "appendix_table1.tex", booktabs
	label
	cells("b (fmt(%9.0fc)) frac (fmt(a2))") 
	unstack 
	nonotes 
	nonumber 
	noobs 
	nomtitle
	drop("Total") 
	collabels("\multicolumn{1}{c}{Households}" "\multicolumn{1}{c}{\% of Baseline}") 
	varlabels(`e(labels)')
	fragment
	replace;

#delimit cr 




