library(readr)
library(rvest)
library(tidyverse)
library(labelled)
library(glue)
library(haven)

rm(list = ls())


##### house_data #####
### grabbing data for households
house_data <- read_dta("~/Documents/academic/academic_projects/conditional_cash_transfers/world_bank_cct/malawi_raw/baseline_2007/sihr1_pi_s2_public.dta") %>% 
  mutate(across(c(s2q02,s2q03, s2q04, s2q05, s2q07, s2q11, s2q12),~as_factor(.x, levels = "label"))) %>% 
  rename(dwelling_type=s2q02, wall_type = s2q03, roof_type = s2q04, floor_type = s2q05, cooking_fuel_type = s2q07, water_type = s2q11,toilet_type = s2q12) %>% 
  select(ea, hhid, respid, contains("type"))

##### respondent_data #####
respondent_data <- read_dta("~/Documents/academic/academic_projects/conditional_cash_transfers/world_bank_cct/malawi_raw/baseline_2007/sihr1_pi_s1_public.dta") %>% 
  mutate(s1q04 = if_else(s1q04 == 2, 0, s1q04) ) %>% 
  mutate(s1q04 = factor(s1q04, levels = c(1, 0),
                      labels = c("Male", "Female"))) %>% 
  mutate(across(c(s1q05,s1q08),~as_factor(.x, levels = "label"))) %>% 
  rename(gender_r1=s1q04, relation_head_r1 = s1q05, age_r1 =  s1q07, highest_ed_r1 = s1q08, years_highest_ed_r1 = s1q08_years) %>% 
  select(ea, hhid, respid, contains("r1"), s1q03, corespid=s1q01)
  
##### respondent_father_ed_data #####
### grabbing parent education data 
parent_ed_data <- read_dta("~/Documents/academic/academic_projects/conditional_cash_transfers/world_bank_cct/malawi_raw/baseline_2007/sihr1_pii_s8_public.dta") %>% 
  mutate(across(c(s8q07a, s8q14a),~as_factor(.x, levels = "label"))) %>% 
  rename(father_highest_ed_r1=s8q07a, father_id_r1 =  s8q02, mother_highest_ed_r1 =  s8q14a, mother_id_r1 =  s8q09) %>% 
  select(ea, hhid, corespid, respid=p1respid, contains("r1"))
  

##### sihr1_identifers  #####
### grabbing identifiers baseline

sihr1_identifers <- read_dta("~/Documents/academic/academic_projects/conditional_cash_transfers/world_bank_cct/malawi_raw/baseline_2007/sihr1_identifers.dta") %>% 
  mutate(across(c(ta_name, cycle_details),~as_factor(.x, levels = "label")))

##### sihr4__identifiers_clean.dta  #####
### grabbing linkage data

link <- read_dta("~/Documents/academic/academic_projects/conditional_cash_transfers/world_bank_cct/malawi_raw/round3_2012/sihr4__identifiers_clean.dta")


##### attending_school outcomes  #####
### grabbing outcomes
attending_school_outcomes <- read_dta("~/Documents/academic/academic_projects/conditional_cash_transfers/world_bank_cct/malawi_raw/round3_2012/SIHR4_PII_S7A_public.dta") %>% 
  mutate(across(c(s7q05, s7q09),~as_factor(.x, levels = "label"))) %>% 
  rename(attending_school_outcome=s7q05, highest_ed_outcome =  s7q09, highest_grade_completed_outcome =  s7q07a) %>% 
  select(ea_r4ex, ea_r4, corespid, hhid_r4, contains("outcome")) 
  

##### competency #####
### grabbing competency data (if we want it)
  competency <- read_dta("~/Documents/academic/academic_projects/conditional_cash_transfers/world_bank_cct/malawi_raw/round3_2012/SIHR4_PII_S11B_public.dta")




  



##### r1 #####
### combining baseline data
r1 <- sihr1_identifers %>% 
  left_join(house_data) %>% 
  left_join(parent_ed_data) %>% 
  left_join(respondent_data %>% filter(s1q03 == 1)) %>% 
  rename(ea_R1=ea, hhid_R1=hhid, corespid_R1=corespid) %>% 
  left_join(link)
##### r4 #####
### combining round 4 and identifier linkage data
r4 <- attending_school_outcomes %>% 
  rename(corespid_R2=corespid, ea_R1=ea_r4  ) %>% 
  left_join(link)

##### final Malawi #####
### joining r1 and r4
final_malawi <- r1 %>% 
  left_join(r4)

saveRDS(final_malawi, "final_malawi.rds")




