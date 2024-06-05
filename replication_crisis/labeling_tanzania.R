library(readr)
library(rvest)
library(tidyverse)
library(labelled)
library(glue)
rm(list = ls())



tanzania_file_name_links <- tibble(file_number = read_html("https://microdata.worldbank.org/index.php/catalog/2669/data-dictionary/F5?") %>% 
  html_element("#tabs-1 > div.tab-body-no-sidebar-x > div > div.col-sm-2.col-md-2.col-lg-2.tab-sidebar.hidden-sm-down.sidebar-files > ul") %>%  
  html_nodes("a") %>%
  html_attr("href") %>% 
  str_extract("(?<=/F)\\d+"), 
  file_name = read_html("https://microdata.worldbank.org/index.php/catalog/2669/data-dictionary/F5?") %>% 
    html_element("#tabs-1 > div.tab-body-no-sidebar-x > div > div.col-sm-2.col-md-2.col-lg-2.tab-sidebar.hidden-sm-down.sidebar-files > ul") %>%  
    html_nodes("a") %>%
    html_attr("href") %>% 
    str_extract("(?<=file_name=)[^&\"]+") %>% 
    str_replace_all(fixed(" "), ""))
  

tanzania_files_csv <- list.files("~/Documents/academic/academic_projects/conditional_cash_transfers/world_bank_cct/tanzania_raw", pattern = ".csv$", recursive = TRUE)
tanzania_file_name_links <- tibble(file_csv = tanzania_files_csv) %>% 
  mutate(file_name = str_remove_all(file_csv, "round1|round2|round3|\\/|\\.csv")) %>% 
  full_join(tanzania_file_name_links)
#files downloaded matched files with links which is good
#loading all files
tanzania_all_df <- map(tanzania_file_name_links$file_csv,~read_csv(glue("~/Documents/academic/academic_projects/conditional_cash_transfers/world_bank_cct/tanzania_raw/{.x}")))

names(tanzania_all_df) <- tanzania_file_name_links$file_name


build_labels_df <- function(number, file) {
  print(glue("https://microdata.worldbank.org/index.php/catalog/2669/data-dictionary/F{number}?file_name={file}"))
  labels_df<- read_html(glue("https://microdata.worldbank.org/index.php/catalog/2669/data-dictionary/F{number}?file_name={file}")) %>% 
    html_element("#variables-container > div.container-fluid.table-variable-list.data-dictionary") %>%
    html_text()  %>% 
    str_split(pattern = "\r\n                                                        \r\n                                \r\n                        \r\n                            \r\n                            ") %>% 
    as_vector() %>% 
    map( ~str_split(.x, pattern = "\r\n                            \r\n                        \r\n                        \r\n                            \r\n                                \r\n                                    ")) %>% 
    map_dfr(~tibble(variable_name = .x[[1]][1], description = str_squish(.x[[1]][2]))) %>% 
    drop_na()
  
  Sys.sleep(rnorm(1, mean = 7, 2))
  return(labels_df)
}



possibly_build_labels <- possibly(build_labels_df)
tanzania_variable_labels <- map(seq(1, nrow(tanzania_file_name_links)), ~possibly_build_labels(tanzania_file_name_links$file_number[.x], tanzania_file_name_links$file_name[.x]) )  
names(tanzania_variable_labels) <-  tanzania_file_name_links$file_name

build_labels_list <- function(description, variable_name){
  var_labels <- description
  names(var_labels) <- variable_name
  var_labels<- as.list(var_labels)
  return(var_labels)
}

tanzania_labels_list <- map(tanzania_variable_labels, ~build_labels_list(.x$description,.x$variable_name))


for (x in 1:length(tanzania_all_df)) {
  var_label(tanzania_all_df[[x]]) <- tanzania_labels_list[[x]]
}

tanzania_list <- list(tanzania_all_df, tanzania_file_name_links, tanzania_labels_list, tanzania_variable_labels)
saveRDS(tanzania_list, "tanzania_list.rds" )

