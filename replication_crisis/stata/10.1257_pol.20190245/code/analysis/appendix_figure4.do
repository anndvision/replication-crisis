
*** 1. INTRODUCTION ***
/*
Description: Tabulates sub-district implementation status by baseline assignment
Uses: pkhben2013_use.data
Creates: appendix_figure4.tex		
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


*** 3. CALCULATE PKH RECEIPT BY KECAMATAN ***
cd "`pkhdata'"
use "Wave I/Data/pkhben2013_use.dta", clear

//keep one observation per kecamatan
bysort ea: keep if _n == 1
//keep only treatment kecs 
drop if missing(L07)

//
gen pkh09_indic = 1 if K09 == 1
gen pkh13_indic = 1 if K13 == 1


*** 4. TABULATE AND EXPORT TO LATEX ***
cd "`latex'"

//midline, control
estpost sum pkh09_indic if L07 == 0
eststo A

sum K09 if L07 == 0
local pctmean = 100*round(`r(mean)', .001)
local pctmeanstr = "`pctmean'"
local percent = "(" + substr("`pctmeanstr'", 1, 4) + "\%)"
estadd local percent "`percent'"

//midline, treatment 
estpost sum pkh09_indic if L07 == 1
eststo B

sum K09 if L07 == 1
local pctmean = 100*round(`r(mean)', .001)
local pctmeanstr = "`pctmean'"
local percent = "(" + substr("`pctmeanstr'", 1, 4) + "\%)"
estadd local percent "`percent'"

//output to latex 
#delimit ; 
esttab A B using "appendix_figure4.tex", booktabs
			label
			cells(count (fmt(0))) 
			unstack 
			nonotes 
			nonumber 
			noobs 
			mtitles("Control (\$ n=180\$)" "Treatment (\$ n=180\$)") 
			mgroups("Baseline Randomization", pattern(1 0) span prefix(\multicolumn{@span}{c}{) suffix(}) erepeat(\cmidrule(lr){@span}))
			collabels(none) 
			varlabels(pkh09_indic "\midrule Treated 2-Year")
			scalars("percent  \ ")
			fragment
			nogaps
			nolines
			replace;
#delimit cr


//endline, control 
estpost sum pkh13_indic if L07 == 0
eststo C

sum K13 if L07 == 0
local pctmean = 100*round(`r(mean)', .001)
local pctmeanstr = "`pctmean'"
local percent = "(" + substr("`pctmeanstr'", 1, 4) + "\%)"
estadd local percent "`percent'"


//endline, treatment 
estpost sum pkh13_indic if L07 == 1
eststo D

sum K13 if L07 == 1
local pctmean = 100*round(`r(mean)', .001)
local pctmeanstr = "`pctmean'"
local percent = "(" + substr("`pctmeanstr'", 1, 4) + "\%)"
estadd local percent "`percent'"

//output to latex 
#delimit ; 
esttab C D using "appendix_figure4.tex", booktabs
			label
			cells(count (fmt(0))) 
			unstack 
			nonotes 
			nonumber 
			noobs 
			nomtitles
			collabels(none) 
			varlabels(pkh13_indic "\\ Treated 6-Year ")
			scalars("percent \ ")
			fragment
			nolines
			nogaps
			append;
#delimit cr




