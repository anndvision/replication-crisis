
*** 1. INTRODUCTION ***
/*
Description: 	Generates .tex file for Appendix Figure 2 (Treatment breakdown by province)
Uses: 			household_allwaves_master.dta
Creates: 		appendix_figure2.tex
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
		global PKH = "$PKH/"	
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


//keep only one observation per kecamatan
bysort province kecamatan: gen nth = _n
keep if nth == 1
drop nth 

//need to label provinces correctly
label define provincelabels 31 "DKI Jakarta" 32 "West Java" 35 "East Java" 53 "East Nusa Tenggara" 71 "North Sulawesi" 75 "Gorontalo"
label values province provincelabels

//tabulate and store
estpost tab province L07 
eststo A


*** 4. OUTPUT TO LATEX ***
cd "`latex'"

#delimit ;

esttab A using "appendix_figure2.tex", booktabs
	label
	cells(b (fmt(0))) 
	unstack 
	nonotes 
	nonumber 
	noobs 
	nomtitle 
	collabels(none) 
	eqlabels(, lhs("Province")) 
	varlabels(`e(labels)', blist(Total "\midrule "))
	fragment
	replace;

#delimit cr 




