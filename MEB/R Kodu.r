library(shiny)
library(readxl)
library(writexl)
library(tibble)
library(openxlsx)
library(shinydashboard)
library(shinyjs)
library(lubridate)
library(ggplot2)
library(tidyr)
library(dplyr)
library(shinythemes)
library(shinydashboardPlus)
library(ggthemes)
library(fresh)
library(readxl)

mytheme <- create_theme(
  adminlte_color(
    light_blue = "#1b1867"
  ),
  adminlte_sidebar(
    width = "200px",
    dark_bg = "#1b1867",
    dark_hover_bg = "#8c2200",
    dark_color = "#f0f7ff"
  ),
  adminlte_global(
    content_bg = "#f7f7f7",
    box_bg = "#ff3f95", 
    info_box_bg = "#bc40fc"
  )
)

price_turk <- read_xlsx("Türk_nisan_2022.xlsx")
price_ssn <- read_xlsx("SSN_nisan_2022.xlsx")
merge_df <- data.frame(
  "Total Food" = numeric(0),
  "Total NFIs" = numeric(0),
  "Total Rent" = numeric(0),
  "Total Utilities" = numeric(0),
  "Total Education" = numeric(0),
  "Total Health" = numeric(0),
  "Total Transportation" = numeric(0),
  "Total Communucation" = numeric(0))
merge_df1 <- data.frame(
  "Inflation Rate" = numeric(0),
  "Food Inflation Rate" = numeric(0), 
  "Shelter Inflation Rate" = numeric(0), 
  "Health Inflation Rate" = numeric(0), 
  "Transportation Inflation Rate" = numeric(0), 
  "Communucation Inflation Rate" = numeric(0), 
  "Education Inflation Rate" = numeric(0)
)


Tuik <- data.frame()
FOOD <- data.frame()
NFI <- data.frame()
EDUCATION <- data.frame()
HEALTH <- data.frame()
Transportation <- data.frame()
Communucation <- data.frame()
rent <- data.frame()
utilities <- data.frame()
meb_cal <- data.frame()
carpim_sonuclari <- data.frame()
meb_multiplier <- data.frame()
meb <- data.frame()
enflasyon2 <- data.frame()
inf <- data.frame()
baslangic_tarihi <- as.Date("2005-01-01")


ui <- dashboardPage(
  dashboardHeader(
    title = "MEB Calculation Tool"
  ),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Intro", tabName = "intro", icon = icon("house")),
      menuItem("Data Upload", tabName = "data_upload", icon = icon("calculator")),
      menuItem("Graphs", tabName = "results", icon = icon("chart-simple")))),
  dashboardBody(use_theme(mytheme),
                useShinyjs(),
                tabItems(
                  tabItem(
                    tabName = "data_upload",
                    fluidPage(
                      titlePanel("Data Upload"),
                      tabsetPanel(
                        tabPanel("MEB Calculated", tableOutput("veriCikisi1")),
                        tabPanel("MEB Calculation by Items", tableOutput("veriCikisi2")),
                        tabPanel("NFIs", tableOutput("veriCikisi3")),
                        tabPanel("Shelter", tableOutput("veriCikisi4")),
                        tabPanel("Other Items", tableOutput("veriCikisi5"))
                      ),
                      fluidRow(
                        column(6, fileInput("dosyaYukleme1", "TUIK Indexes"),
                          actionButton("islemButonu",
                                       icon("calculator"),
                                       label = "Calculate",
                                       style = "color: #fff; background-color: #337ab7; border-color: #2e6da4;"),
                          downloadButton("MEB_Calculated", icon("download"),
                                         label = "Download MEB Items",
                                         style = "color: #fff; background-color: #5cb85c; border-color: #4cae4c")
                        ),
                        column(
                          6,
                          selectInput("info_dropdown", "Choose an option:", 
                                      choices = c("ESSN", "CESSN", "Ineligible", "Turkish"))
                        )
                      ),
                      veriCikisi1 = tableOutput("veriCikisi1"),
                      veriCikisi2 = tableOutput("veriCikisi2"),
                      veriCikisi3 = tableOutput("veriCikisi3"),
                      veriCikisi4 = tableOutput("veriCikisi4"),
                      veriCikisi5 = tableOutput("veriCikisi5")
                    )
                  ),
                  tabItem(
                    tabName = "results",
                    fluidPage(
                      tabsetPanel(
                        tabPanel("MEB Components and Inflation Rate", tableOutput("deneme"))
                      ),
                      sidebarLayout(
                        sidebarPanel(
                          selectInput("selectedVar", "Select MEB Components", choices = colnames(merge_df), multiple = TRUE),
                          selectInput("selectedVar1", "Select Inflation Rate", choices = colnames(merge_df1), multiple = TRUE),
                          uiOutput("dateSlider"),
                          actionButton("submitButton", "Plot")
                        ),
                        mainPanel(
                          plotOutput("myPlot"),
                          plotOutput("myPlot1"),
                          plotOutput("myPlot2")
                        )
                      )
                    )
                  ),
                  tabItem(
                    tabName = "intro",
                    fluidPage(
                      div(
                        class = "black-bg",
                        style = "background-color: #f7f7f7; padding: 15px; border: 1px solid #ddd; border-radius: 5px;",
                        column(2.5,
                               tags$a(href = "https://platform.kizilaykart.org/en/RAPOR.HTML", 
                                      tags$img(src = "kizilay1.jpg", width = 275, height = 275)),
                               tags$a(href = "https://platform.kizilaykart.org/en/Doc/rapor/MEB_report.pdf", 
                                      tags$img(src = "meb3.png", width = 225, height = 275)),
                               tags$a(href = "https://platform.kizilaykart.org/en/Doc/rapor/MEB_September_2023.pdf", 
                                      tags$img(src = "meb1.png", width = 225, height = 275)),
                               tags$p("Click on the images for more information.")
                        ),
                        column(3,
                               div(
                                 style = "background-color: #f7f7f7; padding: 15px; border: 1px solid #ddd; border-radius: 5px;",
                                 HTML("
             <h1 style='margin-top:12.0pt;margin-right:0cm;margin-bottom:.0001pt;margin-left:0cm;font-size:21px;font-family:\"Calibri Light\",sans-serif;color:#2E74B5;font-weight:normal;text-align:center;'><span style='color:windowtext;'><strong><span style='color: rgb(255, 255, 255); background-color: rgb(192, 0, 0);'>Minimum Expenditure Basket (MEB)</span></strong></span></h1>
<p><span style='background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);'>The term 'minimum expenditure basket' refers to the essential set of goods and services considered necessary to maintain a basic standard of living. The concept is often used in the context of poverty measurement and social policy. The basket includes items such as food, shelter, clothing, health care, and education.</span></p>
<p><span style='background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);'>Governments and organizations can use the concept of a minimum expenditure basket to determine the minimum income or resources required for individuals or households to meet their basic needs. The idea is to establish a baseline for poverty calculations and to guide the development of social welfare programs and policies.</span></p>
<p><span style='background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);'>The specific items included in a minimum expenditure basket may vary according to factors such as location, cultural norms, and societal expectations. It is essentially a tool for assessing the cost of living and ensuring that individuals and families have access to the basic goods and services necessary for a decent quality of life.</span></p>               
")
                               )
                        ),
                        column(3, 
                               div(
                                 style = "background-color: #f7f7f7; padding: 15px; border: 1px solid #ddd; border-radius: 5px;",
                                 HTML("<h1 style='margin-top:12.0pt;margin-right:0cm;margin-bottom:.0001pt;margin-left:0cm;font-size:21px;font-family:\"Calibri Light\",sans-serif;color:#2E74B5;font-weight:normal;text-align:center;'><strong><span style='color: rgb(255, 255, 255); background-color: rgb(192, 0, 0);'>MEB Components</span></strong></h1>
<ul>
    <li><span style='background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);'>Shelter</span></li>
    <li><span style='background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);'>Wash (water, sanitation, and hygiene)</span></li>
    <li><span style='background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);'>Clothing</span></li>
    <li><span style='background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);'>Education</span></li>
    <li><span style='background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);'>Health</span></li>
    <li><span style='background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);'>Livelihoods</span></li>
    <li><span style='background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);'>Tax and community contribution</span></li>
    <li><span style='background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);'>Protection and security</span></li>
    <li><span style='background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);'>Healthy diet food basket</span></li>
</ul>
<p><span style='background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);'>Compliance with Sphere standards has been taken into account in the preparation of the MEB. In the T&uuml;rkiye context, six of the groups that make up the MEB are included: food, shelter, education, health, protection, and communication. This basket consists mainly of food and shelter. The food basket is prepared for a balanced diet and a minimum daily energy of 2100 kcal and is based on the average demographic structure of households.</span></p>
"))),
                        column(6,
                               div(
                                 style = "background-color: #f7f7f7; padding: 15px; border: 1px solid #ddd; border-radius: 5px;",
                                 HTML(
                                   "<p>&nbsp;</p>
<table style=‘border: none;border-collapse: collapse;width:559pt;’>
    <tbody>
        <tr>
            <td style='color:white;font-size:13px;font-weight:700;font-style:normal;text-decoration:underline;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background:#C00000;height:14.4pt;width:79pt;'>Food Basket</td>
            <td style='color:white;font-size:13px;font-weight:700;font-style:normal;text-decoration:underline;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background:#C00000;width:75pt;'>Education</td>
            <td style='color:white;font-size:13px;font-weight:700;font-style:normal;text-decoration:underline;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background:#C00000;width:131pt;'>Health (3 visits)</td>
            <td style='color:white;font-size:13px;font-weight:700;font-style:normal;text-decoration:underline;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background:#C00000;width:70pt;'>Shelter</td>
            <td style='color:white;font-size:13px;font-weight:700;font-style:normal;text-decoration:underline;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background:#C00000;width:133pt;'>Hygiene</td>
            <td style='color:white;font-size:13px;font-weight:700;font-style:normal;text-decoration:underline;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background:#C00000;width:71pt;'>Protection</td>
        </tr>
        <tr>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);height:14.4pt;'>Rice</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'>Notebook</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'>Medicines</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'>Actual Rent</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'>Detergents (For Laundry)</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'>Transport</td>
        </tr>
        <tr>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);height:14.4pt;'>Bulgur</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'>Pencil</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'>Fees Paid To Specialist Doctor</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'>Water&nbsp;</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'>Dishwasher Detergents</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'>Communication</td>
        </tr>
        <tr>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);height:14.4pt;'>Bread</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'>Other stationery</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'>Electricity</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'>Disinfectants And Insecticidies</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
        </tr>
        <tr>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);height:14.4pt;'>Yoghurt</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'>Tube Gas (12 L)</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'>Shaving Articles</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
        </tr>
        <tr>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);height:14.4pt;'>White Cheese</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'>Articles For Dental Hygiene</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
        </tr>
        <tr>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);height:14.4pt;'>Egg</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'>Bath Soap</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
        </tr>
        <tr>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);height:14.4pt;'>Sun-Flower Oil</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'>Hair Care Products</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
        </tr>
        <tr>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);height:14.4pt;'>Tomato</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'>Toilet Paper</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
        </tr>
        <tr>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);height:14.4pt;'>Cucumber</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'>Baby Napkin</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
        </tr>
        <tr>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);height:14.4pt;'>Dry Bean</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'>Hygiene Pad For Women</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
        </tr>
        <tr>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);height:14.4pt;'>Granulated Sugar</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
        </tr>
        <tr>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);height:14.4pt;'>Salt</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:general;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
        </tr>
        <tr>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:justify;vertical-align:middle;border:none;background: rgb(244, 244, 244);height:14.4pt;'>Tea</td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:justify;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:justify;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:justify;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:justify;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
            <td style='color:black;font-size:13px;font-weight:400;font-style:normal;text-decoration:none;font-family:’Calibri Light’, sans-serif;text-align:justify;vertical-align:middle;border:none;background: rgb(244, 244, 244);'><br></td>
        </tr>
    </tbody>
</table>
<p><br></p>
"
                                 )
                               )
                        )
                      ),
                      tags$style(HTML("
      .black-bg {
        overflow: hidden; /* Eklediğim stil */
      }
    ")
                      )
                    )
                  )
                )))

server <- function(input, output, session) {
  observeEvent(input$info_dropdown, {
    selected_option <- input$info_dropdown
    
    if (selected_option == "ESSN") {
      veri1 <- reactive({
        req(input$dosyaYukleme1)
        dosya <- input$dosyaYukleme1
        if (is.null(dosya))
          return(NULL)
        veri <- read_xlsx(dosya$datapath, skip = 4)
        return(veri)
      })
      
      date3 <- reactive({
        inf <- veri1()
        colnames(inf)[1:3] <- c("Yıl", "Ay", "Months")
        inf <- head(inf, -5)
        
        enflasyon2 <- inf[-c(1, 2), ]
        ilk_uc_sutun <- enflasyon2[, 1:3]
        
        secilen_sutunlar <- c("01111", "01113", "01146", "01145", "01148", "01151", "01172", "01176", 
                              "01181", "01193", "01230", "09740", "06111", "06231", "04110", "04411", 
                              "04510", "04522", "05611", "13120", "08320", "07321")
        
        enflasyon_fin <- enflasyon2[, secilen_sutunlar]
        
        Tuik <- cbind(ilk_uc_sutun, enflasyon_fin)
        
        
        baslangic_tarihi <- as.Date("2005-01-01")
        
        date <- seq(baslangic_tarihi, by = "1 month", length.out = nrow(Tuik))
        
        return(date)
      })
      
      output$dateSlider <- renderUI({
        sliderInput("dateSlider", "Select Date Range", 
                    min = as.Date("2025-12-01"), 
                    max = max(date3()), 
                    value = c(as.Date("2025-12-01"), max(date3())),
                    step = 30,
                    timeFormat = "%Y-%m-%d")
      })
      
      observeEvent(input$islemButonu, {
        inf <- veri1()
        colnames(inf)[1:3] <- c("Yıl", "Ay", "Months")
        inf <- head(inf, -5)
        enflasyon2 <- inf[-c(1, 2), ]
        ilk_uc_sutun <- enflasyon2[, 1:3]

        secilen_sutunlar <- c("01111", "01113", "01146", "01145", "01148", "01151", "01172", "01176", 
                              "01181", "01193", "01230", "09740", "06111", "06231", "04110", "04411", 
                              "04510", "04522", "05611", "13120", "08320", "07321")      
        enflasyon_fin <- enflasyon2[, secilen_sutunlar]
        
        Tuik <<- cbind(ilk_uc_sutun, enflasyon_fin)
        
        
        baslangic_tarihi <- as.Date("2005-01-01")
        
        
        date <- seq(baslangic_tarihi, by = "1 month", length.out = nrow(Tuik))
        Date <- as.data.frame(date)
        date3 <- tail(Date, -251)
        Date1 <- tail(Date, -1)
        enf_sutun <- c("0", "011", "04", "06", "07", "08", "10", "13")
        enf_index <- enflasyon2[, enf_sutun]
        colnames(enf_index) <- c("Inflation.Rate", "Food.Inflation.Rate", "Shelter.Inflation.Rate", "Health.Inflation.Rate",
                                 "Transportation.Inflation.Rate", "Communucation.Inflation.Rate", "Education.Inflation.Rate", "NFIs")
        
        enf_index <- enf_index %>% mutate_all(as.numeric)
        enf_rate <- data.frame()
        for (i in 1:nrow(enf_index)) {
          for (j in 1:ncol(enf_index)) {
            if (i > 12 && enf_index[i, j] != 0) {
              enf_rate[i, j] <- (enf_index[i, j] / enf_index[i - 12, j] -1)
            } else {
              enf_rate[i, j] <- NA
            }
          }
        }
        enf_rate <- tail(enf_rate, -251)
        colnames(enf_rate) <- c("Inflation.Rate", "Food.Inflation.Rate", "Shelter.Inflation.Rate", "Health.Inflation.Rate",
                                "Transportation.Inflation.Rate", "Communucation.Inflation.Rate", "Education.Inflation.Rate", "NFIs")
        
        # GIDA VE ALKOLSÜZ İÇECEKLER
        pirinç <- as.numeric(Tuik$"01111")
        bulgur <- as.numeric(Tuik$"01111")
        ekmek <- as.numeric(Tuik$"01113")
        yogurt <- as.numeric(Tuik$"01146")
        peynir <- as.numeric(Tuik$"01145")
        yumurta <- as.numeric(Tuik$"01148")
        yag <- as.numeric(Tuik$"01151") 
        domates <- as.numeric(Tuik$"01172")
        salatalık <- as.numeric(Tuik$"01172")
        fasulye <- as.numeric(Tuik$"01176") 
        şeker <- as.numeric(Tuik$"01181")
        tuz <- as.numeric(Tuik$"01193")
        çay <- as.numeric(Tuik$"01230")
        
        # EĞİTİM VE KIRTASİYE
        defter <- as.numeric(Tuik$"09740")
        kalem <- as.numeric(Tuik$"09740")
        kırtasiye <- as.numeric(Tuik$"09740")
        
        # SAĞLIK VE KONUT
        ilaç <- as.numeric(Tuik$"06111")
        doktor <- as.numeric(Tuik$"06231")
        kira <- as.numeric(Tuik$"04110")
        su <- as.numeric(Tuik$"04411")
        elektrik <- as.numeric(Tuik$"04510")
        gaz <- as.numeric(Tuik$"04522")
        
        # TEMİZLİK VE KİŞİSEL BAKIM
        çamasır_det <- as.numeric(Tuik$"05611")
        bulasık_det <- as.numeric(Tuik$"05611")
        dezenfektan <- as.numeric(Tuik$"05611")
        
        # Kişisel bakım ürünleri genel olarak 12132 kodu altında toplanmıştır
        tıraş <- as.numeric(Tuik$"13120")
        agız_bakım <- as.numeric(Tuik$"13120")
        sabun <- as.numeric(Tuik$"13120")
        şampuan <- as.numeric(Tuik$"13120")
        tuvalet_kagıdı <- as.numeric(Tuik$"13120")
        bebek_bezi <- as.numeric(Tuik$"13120")
        hijyenik_ped <- as.numeric(Tuik$"13120")
        
        # ULAŞTIRMA VE HABERLEŞME
        telefon <- as.numeric(Tuik$"08320")
        otobüs <- as.numeric(Tuik$"07321")
        
        
        meb <- as.data.frame(cbind(pirinç, bulgur, ekmek, yogurt, peynir, yumurta, yag, domates, salatalık, fasulye, şeker,tuz, çay, defter, kalem, kırtasiye, ilaç, doktor, kira, su,
                                   elektrik, gaz, çamasır_det, bulasık_det, dezenfektan, tıraş, agız_bakım, sabun, şampuan, tuvalet_kagıdı, bebek_bezi, hijyenik_ped, telefon,otobüs))
        meb <- cbind(Date, meb)
        meb_new <- tail(meb, -251)
        meb_multiplier <- data.frame()
        
        for (i in 1:nrow(meb)) {
          for (j in 1:ncol(meb)) {
            if (i == 1) {
              meb_multiplier[i, j] <- NA
            } else if (is.numeric(meb[, j])) {
              meb_multiplier[i, j] <- meb[i, j] / meb[i - 1, j]
            }
          }
        }
        
        meb_multiplier <- tail(meb_multiplier, -1)
        meb_multiplier <- meb_multiplier[, -1]
        
        meb_multiplier <- cbind(Date1, meb_multiplier)
        meb_multiplier <- tail(meb_multiplier, -251)
        
        
        colnames(meb_multiplier) <- colnames(price_ssn)
        
        carpim_sonuclari <- meb_multiplier
        
        
        for (i in 1:ncol(meb_multiplier)) {
          if (is.numeric(meb_multiplier[, i])) {
            if (nrow(meb_multiplier) == 1) {
              carpim_sonuclari[1, i] <- meb_multiplier[1, i]
            } else {
              for (j in 2:nrow(meb_multiplier)) {
                carpim_sonuclari[j, i] <- meb_multiplier[j, i] * carpim_sonuclari[j - 1, i]
              }
            }
          }
        }
        
        price_ssn <- price_ssn[, -1]
        carpim_sonuclari <- carpim_sonuclari[, -1]
        result <- data.frame()
        
        for (i in 1:nrow(carpim_sonuclari)) {
          row <- carpim_sonuclari[i, ] * price_ssn
          result <- rbind(result, row)
        }
        
        
        result <- rbind(price_ssn, result)
        meb_cal <- cbind(date3, result)
        
        meb_multi <- data.frame(
          pirinç = 18.9,
          bulgur = 9.45,
          ekmek = 47.25,
          yogurt = 9.45,
          peynir = 9.45,
          yumurta = 189,
          yag = 4.725,
          domates = 5.67,
          salatalık = 5.67,
          fasulye = 9.45,
          şeker = 9.45,
          tuz = 0.945,
          çay = 0.945,
          defter = 2,
          kalem = 2,
          kırtasiye = 2,
          ilaç = 3,
          doktor = 3,
          kira = 1,
          su = 15,
          elektrik = 208.333,
          gaz = 1,
          çamasır_det = 1.5,
          bulaşık_det = 0.75,
          dezenfektan = 0.5,
          tıraş = 2,
          ağız_bakım = 1,
          sabun = 1.5,
          şampuan = 0.65,
          tuvalet_kâğıdı = 12,
          bebek_bezi = 150,
          hijyenik_ped = 30,
          telefon = 1,
          otobüs = 32
        )
        
        meb_cal <- meb_cal[, -1]
        
        colnames(meb_multi) <- colnames(meb_cal)
        
        meb_final <- data.frame()
        for (i in seq_len(nrow(meb_cal))) {
          row <- meb_cal[i, ]
          num_cols <- sapply(row, is.numeric)
          row[num_cols] <- row[num_cols] * meb_multi[num_cols]
          meb_final <- rbind(meb_final, row)
        }
        
        meb_cal <- cbind(date3, meb_final)
        
        gıda <- c("date", "Rice", "Bulgur", "Bread", "Yogurt", "White cheese",
                  "Egg", "Sunflower oil", "Tomatoes", "Cucumber", "Dry beans", "Sugar",
                  "Salt", "Tea")
        non_food <- c("date", "Laundry detergent", "Dishwashing liquid", "Disinfectant", "Shaving articles",
                      "Dental hygiene", "Soap", "Haircare", "Toilet paper", "Diaper", "Hygiene Pad for Women")
        edu <- c("date", "Notebook", "Pencil", "Other stationary")
        hp <- c("date", "Medicine", "Specialist")
        transport <- c("date", "Public transportation")
        commun <- c("date", "Mobile service package")
        kira <- c("date", "Rent")
        fatura <- c("date", "Water", "Electricity", "Gas canister (12 L)")
        barinma <- c("date", "Rent", "Water", "Electricity", "Gas canister (12 L)")
        diger <- c("date", "Notebook", "Pencil", "Other stationary", "Medicine", "Specialist", "Public transportation", "Mobile service package")
        
        FOOD <- meb_cal[,gıda]
        NFI <- meb_cal[, non_food]
        EDUCATION <- meb_cal[, edu]
        HEALTH <- meb_cal[, hp]
        Transportation <- meb_cal[, transport]
        Communucation <- meb_cal[, commun]
        rent <- meb_cal[, kira]
        utilities <- meb_cal[, fatura]
        shelter <- meb_cal[, barinma]
        other_items <- meb_cal[, diger]
        
        FOOD$Food.Total <- rowSums(FOOD[, c("Rice", "Bulgur", "Bread", "Yogurt", "White cheese", "Egg", "Sunflower oil", "Tomatoes", "Cucumber", "Dry beans", "Sugar", "Salt", "Tea")])
        NFI$NFI.Total <- rowSums(NFI[, c("Laundry detergent", "Dishwashing liquid", "Disinfectant", "Shaving articles",
                                         "Dental hygiene", "Soap", "Haircare", "Toilet paper", "Diaper", "Hygiene Pad for Women")])
        EDUCATION$Education.Total <- rowSums(EDUCATION[, c("Notebook", "Pencil", "Other stationary")])
        HEALTH$Health.Total <- rowSums(HEALTH[, c("Medicine", "Specialist")])
        Transportation$Transportation.Total <- Transportation$`Public transportation`
        Communucation$Communucation.Total <- Communucation$`Mobile service package`
        rent$Rent.Total <- rent$Rent
        utilities$Utilities.Total <- rowSums(utilities[, c("Water", "Electricity", "Gas canister (12 L)")])
        shelter$Total <- rowSums(shelter[, c("Rent", "Water", "Electricity", "Gas canister (12 L)")])
        other_items$Total <- rowSums(other_items[, c("Notebook", "Pencil", "Other stationary", "Medicine", "Specialist", "Public transportation", "Mobile service package")])
        
        
        NFI1 <- NFI[, -1]
        EDUCATION1 <- EDUCATION[, -1]
        HEALTH1 <- HEALTH[, -1]
        Transportation1 <- Transportation[, -1]
        Communucation1 <-Communucation[, -1]
        rent1 <-rent[, -1]
        utilities1 <-utilities[, -1]
        
        
        merge_df <<- as.data.frame(cbind(FOOD$Food.Total, NFI1$NFI.Total, rent1$Rent.Total, utilities1$Utilities.Total, EDUCATION1$Education.Total, 
                                         HEALTH1$Health.Total, Transportation1$Transportation.Total, Communucation1$Communucation.Total))
        
        colnames(merge_df) <<- c("Total.Food", "Total.NFIs", "Total.Rent", "Total.Utilities", "Total.Education", "Total.Health", "Total.Transportation", "Total.Communucation")
        merge_df <<- cbind(date3, merge_df)
        merge_df$MEB.Total <- rowSums(merge_df[, c("Total.Food", "Total.NFIs", "Total.Rent", "Total.Utilities", "Total.Education", "Total.Health", "Total.Transportation", "Total.Communucation")])
        merge_df1 <<- cbind(merge_df, enf_rate)
        
        
        all_components <- cbind(FOOD, NFI1, rent1, utilities1, EDUCATION1, HEALTH1, Transportation1, Communucation1)
        
        wb <- createWorkbook()
        addWorksheet(wb, "Veri1")
        writeData(wb, sheet = "Veri1", x = merge_df)
        
        addWorksheet(wb, "Veri2")
        writeData(wb, sheet = "Veri2", x = all_components)
        
        addWorksheet(wb, "Veri3")
        writeData(wb, sheet = "Veri3", x = meb_new)
        
        addWorksheet(wb, "Veri4")
        writeData(wb, sheet = "Veri4", x = meb_multiplier)
        
        saveWorkbook(wb, file = "MEB_Calculation.xlsx", overwrite = TRUE)
        
        output$veriCikisi1 <- renderTable({
          merge_df$date <- as.character(merge_df$date)
          merge_df
        })
        
        output$veriCikisi2 <- renderTable({
          FOOD$date <- as.character(FOOD$date)
          FOOD
        })
        
        output$veriCikisi3 <- renderTable({
          NFI$date <- as.character(NFI$date)
          NFI
        })
        
        output$veriCikisi4 <- renderTable({
          shelter$date <- as.character(shelter$date)
          shelter
        })
        
        output$veriCikisi5 <- renderTable({
          other_items$date <- as.character(other_items$date)
          other_items
        })
        
        output$MEB_Calculated <- downloadHandler(
          filename = function() {
            "MEB_Calculation.xlsx"
          },
          content = function(file) {
            file.copy("MEB_Calculation.xlsx", file)
          }
        )
        
        observeEvent(input$dateSlider, {
          selectedDateRange <- input$dateSlider
          
          observeEvent(input$submitButton, {
            selectedVariables <- input$selectedVar
            selectedVariables1 <- input$selectedVar1
            filteredData <- merge_df1[, c("date", selectedVariables, selectedVariables1)]
            filteredData <- filteredData[filteredData$date >= selectedDateRange[1] & filteredData$date <= selectedDateRange[2], ]
            meltedData <- tidyr::gather(filteredData, key = "variable", value = "value", -date)
            
            meltedData$variable <- as.character(meltedData$variable)
            meltedData$value <- as.numeric(meltedData$value)
            
            output$myPlot <- renderPlot({
              ggplot(meltedData, aes(x = date)) +
                geom_bar(data = filter(meltedData, variable %in% selectedVariables), stat = "identity", position = "dodge", aes(y = value, fill = variable)) +
                geom_line(data = filter(meltedData, variable %in% selectedVariables1), aes(y = value * 15000, group = variable, color = variable), linewidth = 2) +
                labs(title = "Monthly amount of MEB components and Annual Inflation Rates",
                     x = "Date", y = "MEB Component Value",
                     fill = "MEB Components", color ="CPI") +
                scale_y_continuous(sec.axis = sec_axis(~./15000, name = "Inflation Rates")) +
                scale_fill_manual(values = c("orangered", "forestgreen", "firebrick", "olivedrab4", "lightseagreen", "bisque2", "linen", "plum", "royalblue")) +
                scale_color_manual(values = c("navyblue", "ivory3", "goldenrod", "salmon4", "darkorchid", "burlywood3", "rosybrown", "cornflowerblue")) + 
                theme_stata() + theme(legend.position = "top",
                                      legend.key = element_rect(color = "grey50"),
                                      legend.key.width = unit(0.9, "cm"),
                                      legend.key.height = unit(0.75, "cm")
                )
            })
            
            output$myPlot1 <- renderPlot({
              ggplot(meltedData, aes(x = date)) +
                geom_bar(data = filter(meltedData, variable %in% selectedVariables), stat = "identity", position = "dodge", aes(y = value, fill = variable)) +
                labs(title = "Monthly amount of MEB components",
                     x = "Date", y = "MEB Component Value",
                     fill = "MEB Components") +
                scale_fill_manual(values = c("orangered", "forestgreen", "firebrick", "olivedrab4", "lightseagreen", "bisque2", "linen", "plum", "royalblue")) + 
                theme_stata() + theme(legend.position = "top",
                                      legend.key = element_rect(color = "grey50"),
                                      legend.key.width = unit(0.9, "cm"),
                                      legend.key.height = unit(0.75, "cm")
                )
            })
            
            output$myPlot2 <- renderPlot({
              ggplot(meltedData, aes(x = date)) +
                geom_line(data = filter(meltedData, variable %in% selectedVariables1), aes(y = value, group = variable, color = variable), linewidth = 2) +
                labs(title = "Annual Inflation Rate",
                     x = "Date", y = "Inflation Rate",
                     color = "CPI") +
                scale_color_manual(values = c("navyblue", "ivory3", "goldenrod", "salmon4", "darkorchid", "burlywood3", "rosybrown", "cornflowerblue")) + 
                theme_stata() + theme(legend.position = "top",
                                      legend.key = element_rect(color = "grey50"),
                                      legend.key.width = unit(0.9, "cm"),
                                      legend.key.height = unit(0.75, "cm")
                )
            })
          })
        })
      })
    }
  })
  observeEvent(input$info_dropdown, {
    selected_option <- input$info_dropdown
    
    if (selected_option == "CESSN") {
      veri1 <- reactive({
        req(input$dosyaYukleme1)
        dosya <- input$dosyaYukleme1
        if (is.null(dosya))
          return(NULL)
        veri <- read_xlsx(dosya$datapath, skip = 4)
        return(veri)
      })
      
      date3 <- reactive({
        inf <- veri1()
        colnames(inf)[1:3] <- c("Yıl", "Ay", "Months")
        inf <- head(inf, -5)
        enflasyon2 <- inf[-c(1, 2), ]
        ilk_uc_sutun <- enflasyon2[, 1:3]
        
        secilen_sutunlar <- c("01111", "01113", "01146", "01145", "01148", "01151", "01172", "01176", 
                              "01181", "01193", "01230", "09740", "06111", "06231", "04110", "04411", 
                              "04510", "04522", "05611", "13120", "08320", "07321")
        
        enflasyon_fin <- enflasyon2[, secilen_sutunlar]
        
        Tuik <- cbind(ilk_uc_sutun, enflasyon_fin)
        
        
        baslangic_tarihi <- as.Date("2005-01-01")
        
        date <- seq(baslangic_tarihi, by = "1 month", length.out = nrow(Tuik))
        
        return(date)
      })
      
      output$dateSlider <- renderUI({
        sliderInput("dateSlider", "Select Date Range", 
                    min = as.Date("2025-12-01"), 
                    max = max(date3()), 
                    value = c(as.Date("2025-12-01"), max(date3())),
                    step = 30,
                    timeFormat = "%Y-%m-%d")
      })
      
      observeEvent(input$islemButonu, {
        inf <- veri1()
        colnames(inf)[1:3] <- c("Yıl", "Ay", "Months")
        inf <- head(inf, -5)
        enflasyon2 <- inf[-c(1, 2), ]
        ilk_uc_sutun <- enflasyon2[, 1:3]
        
        secilen_sutunlar <- c("01111", "01113", "01146", "01145", "01148", "01151", "01172", "01176", 
                              "01181", "01193", "01230", "09740", "06111", "06231", "04110", "04411", 
                              "04510", "04522", "05611", "13120", "08320", "07321")      
        enflasyon_fin <- enflasyon2[, secilen_sutunlar]
        
        Tuik <<- cbind(ilk_uc_sutun, enflasyon_fin)
        
        
        baslangic_tarihi <- as.Date("2005-01-01")
        
        
        date <- seq(baslangic_tarihi, by = "1 month", length.out = nrow(Tuik))
        Date <- as.data.frame(date)
        date3 <- tail(Date, -251)
        Date1 <- tail(Date, -1)
        enf_sutun <- c("0", "011", "04", "06", "07", "08", "10", "13")
        enf_index <- enflasyon2[, enf_sutun]
        colnames(enf_index) <- c("Inflation.Rate", "Food.Inflation.Rate", "Shelter.Inflation.Rate", "Health.Inflation.Rate",
                                 "Transportation.Inflation.Rate", "Communucation.Inflation.Rate", "Education.Inflation.Rate", "NFIs")
        
        enf_index <- enf_index %>% mutate_all(as.numeric)
        enf_rate <- data.frame()
        for (i in 1:nrow(enf_index)) {
          for (j in 1:ncol(enf_index)) {
            if (i > 12 && enf_index[i, j] != 0) {
              enf_rate[i, j] <- (enf_index[i, j] / enf_index[i - 12, j] -1)
            } else {
              enf_rate[i, j] <- NA
            }
          }
        }
        enf_rate <- tail(enf_rate, -251)
        colnames(enf_rate) <- c("Inflation.Rate", "Food.Inflation.Rate", "Shelter.Inflation.Rate", "Health.Inflation.Rate",
                                "Transportation.Inflation.Rate", "Communucation.Inflation.Rate", "Education.Inflation.Rate", "NFIs")
        
        # GIDA VE ALKOLSÜZ İÇECEKLER
        pirinç <- as.numeric(Tuik$"01111")
        bulgur <- as.numeric(Tuik$"01111")
        ekmek <- as.numeric(Tuik$"01113")
        yogurt <- as.numeric(Tuik$"01146")
        peynir <- as.numeric(Tuik$"01145")
        yumurta <- as.numeric(Tuik$"01148")
        yag <- as.numeric(Tuik$"01151") 
        domates <- as.numeric(Tuik$"01172")
        salatalık <- as.numeric(Tuik$"01172")
        fasulye <- as.numeric(Tuik$"01176") 
        şeker <- as.numeric(Tuik$"01181")
        tuz <- as.numeric(Tuik$"01193")
        çay <- as.numeric(Tuik$"01230")
        
        # EĞİTİM VE KIRTASİYE
        defter <- as.numeric(Tuik$"09740")
        kalem <- as.numeric(Tuik$"09740")
        kırtasiye <- as.numeric(Tuik$"09740")
        
        # SAĞLIK VE KONUT
        ilaç <- as.numeric(Tuik$"06111")
        doktor <- as.numeric(Tuik$"06231")
        kira <- as.numeric(Tuik$"04110")
        su <- as.numeric(Tuik$"04411")
        elektrik <- as.numeric(Tuik$"04510")
        gaz <- as.numeric(Tuik$"04522")
        
        # TEMİZLİK VE KİŞİSEL BAKIM
        çamasır_det <- as.numeric(Tuik$"05611")
        bulasık_det <- as.numeric(Tuik$"05611")
        dezenfektan <- as.numeric(Tuik$"05611")
        
        # Kişisel bakım ürünleri genel olarak 12132 kodu altında toplanmıştır
        tıraş <- as.numeric(Tuik$"13120")
        agız_bakım <- as.numeric(Tuik$"13120")
        sabun <- as.numeric(Tuik$"13120")
        şampuan <- as.numeric(Tuik$"13120")
        tuvalet_kagıdı <- as.numeric(Tuik$"13120")
        bebek_bezi <- as.numeric(Tuik$"13120")
        hijyenik_ped <- as.numeric(Tuik$"13120")
        
        # ULAŞTIRMA VE HABERLEŞME
        telefon <- as.numeric(Tuik$"08320")
        otobüs <- as.numeric(Tuik$"07321")
        
        
        meb <- as.data.frame(cbind(pirinç, bulgur, ekmek, yogurt, peynir, yumurta, yag, domates, salatalık, fasulye, şeker,tuz, çay, defter, kalem, kırtasiye, ilaç, doktor, kira, su,
                                   elektrik, gaz, çamasır_det, bulasık_det, dezenfektan, tıraş, agız_bakım, sabun, şampuan, tuvalet_kagıdı, bebek_bezi, hijyenik_ped, telefon,otobüs))
        meb <- cbind(Date, meb)
        meb_new <- tail(meb, -251)
        meb_multiplier <- data.frame()
        
        for (i in 1:nrow(meb)) {
          for (j in 1:ncol(meb)) {
            if (i == 1) {
              meb_multiplier[i, j] <- NA
            } else if (is.numeric(meb[, j])) {
              meb_multiplier[i, j] <- meb[i, j] / meb[i - 1, j]
            }
          }
        }
        
        meb_multiplier <- tail(meb_multiplier, -1)
        meb_multiplier <- meb_multiplier[, -1]
        
        meb_multiplier <- cbind(Date1, meb_multiplier)
        meb_multiplier <- tail(meb_multiplier, -251)
        
        
        colnames(meb_multiplier) <- colnames(price_ssn)
        
        carpim_sonuclari <- meb_multiplier
        
        
        for (i in 1:ncol(meb_multiplier)) {
          if (is.numeric(meb_multiplier[, i])) {
            if (nrow(meb_multiplier) == 1) {
              carpim_sonuclari[1, i] <- meb_multiplier[1, i]
            } else {
              for (j in 2:nrow(meb_multiplier)) {
                carpim_sonuclari[j, i] <- meb_multiplier[j, i] * carpim_sonuclari[j - 1, i]
              }
            }
          }
        }
        
        price_ssn <- price_ssn[, -1]
        carpim_sonuclari <- carpim_sonuclari[, -1]
        result <- data.frame()
        
        for (i in 1:nrow(carpim_sonuclari)) {
          row <- carpim_sonuclari[i, ] * price_ssn
          result <- rbind(result, row)
        }
        
        
        result <- rbind(price_ssn, result)
        meb_cal <- cbind(date3, result)
        
        meb_multi <- data.frame(
          pirinç = 12.9,
          bulgur = 6.45,
          ekmek = 32.25,
          yogurt = 6.45,
          peynir = 6.45,
          yumurta = 129,
          yag = 3.225,
          domates = 3.87,
          salatalık = 3.87,
          fasulye = 6.45,
          şeker = 6.45,
          tuz = 0.645,
          çay = 0.645,
          defter = 2,
          kalem = 2,
          kırtasiye = 2,
          ilaç = 3,
          doktor = 3,
          kira = 1,
          su = 15,
          elektrik = 208.333,
          gaz = 1,
          çamasır_det = 1.5,
          bulaşık_det = 0.75,
          dezenfektan = 0.5,
          tıraş = 2,
          ağız_bakım = 1,
          sabun = 1.5,
          şampuan = 0.65,
          tuvalet_kâğıdı = 12,
          bebek_bezi = 150,
          hijyenik_ped = 30,
          telefon = 1,
          otobüs = 32
        )
        
        meb_cal <- meb_cal[, -1]
        
        colnames(meb_multi) <- colnames(meb_cal)
        
        meb_final <- data.frame()
        for (i in seq_len(nrow(meb_cal))) {
          row <- meb_cal[i, ]
          num_cols <- sapply(row, is.numeric)
          row[num_cols] <- row[num_cols] * meb_multi[num_cols]
          meb_final <- rbind(meb_final, row)
        }
        
        meb_cal <- cbind(date3, meb_final)
        
        gıda <- c("date", "Rice", "Bulgur", "Bread", "Yogurt", "White cheese",
                  "Egg", "Sunflower oil", "Tomatoes", "Cucumber", "Dry beans", "Sugar",
                  "Salt", "Tea")
        non_food <- c("date", "Laundry detergent", "Dishwashing liquid", "Disinfectant", "Shaving articles",
                      "Dental hygiene", "Soap", "Haircare", "Toilet paper", "Diaper", "Hygiene Pad for Women")
        edu <- c("date", "Notebook", "Pencil", "Other stationary")
        hp <- c("date", "Medicine", "Specialist")
        transport <- c("date", "Public transportation")
        commun <- c("date", "Mobile service package")
        kira <- c("date", "Rent")
        fatura <- c("date", "Water", "Electricity", "Gas canister (12 L)")
        barinma <- c("date", "Rent", "Water", "Electricity", "Gas canister (12 L)")
        diger <- c("date", "Notebook", "Pencil", "Other stationary", "Medicine", "Specialist", "Public transportation", "Mobile service package")
        
        FOOD <- meb_cal[,gıda]
        NFI <- meb_cal[, non_food]
        EDUCATION <- meb_cal[, edu]
        HEALTH <- meb_cal[, hp]
        Transportation <- meb_cal[, transport]
        Communucation <- meb_cal[, commun]
        rent <- meb_cal[, kira]
        utilities <- meb_cal[, fatura]
        shelter <- meb_cal[, barinma]
        other_items <- meb_cal[, diger]
        
        FOOD$Food.Total <- rowSums(FOOD[, c("Rice", "Bulgur", "Bread", "Yogurt", "White cheese", "Egg", "Sunflower oil", "Tomatoes", "Cucumber", "Dry beans", "Sugar", "Salt", "Tea")])
        NFI$NFI.Total <- rowSums(NFI[, c("Laundry detergent", "Dishwashing liquid", "Disinfectant", "Shaving articles",
                                         "Dental hygiene", "Soap", "Haircare", "Toilet paper", "Diaper", "Hygiene Pad for Women")])
        EDUCATION$Education.Total <- rowSums(EDUCATION[, c("Notebook", "Pencil", "Other stationary")])
        HEALTH$Health.Total <- rowSums(HEALTH[, c("Medicine", "Specialist")])
        Transportation$Transportation.Total <- Transportation$`Public transportation`
        Communucation$Communucation.Total <- Communucation$`Mobile service package`
        rent$Rent.Total <- rent$Rent
        utilities$Utilities.Total <- rowSums(utilities[, c("Water", "Electricity", "Gas canister (12 L)")])
        shelter$Total <- rowSums(shelter[, c("Rent", "Water", "Electricity", "Gas canister (12 L)")])
        other_items$Total <- rowSums(other_items[, c("Notebook", "Pencil", "Other stationary", "Medicine", "Specialist", "Public transportation", "Mobile service package")])
        
        
        NFI1 <- NFI[, -1]
        EDUCATION1 <- EDUCATION[, -1]
        HEALTH1 <- HEALTH[, -1]
        Transportation1 <- Transportation[, -1]
        Communucation1 <-Communucation[, -1]
        rent1 <-rent[, -1]
        utilities1 <-utilities[, -1]
        
        
        merge_df <<- as.data.frame(cbind(FOOD$Food.Total, NFI1$NFI.Total, rent1$Rent.Total, utilities1$Utilities.Total, EDUCATION1$Education.Total, 
                                         HEALTH1$Health.Total, Transportation1$Transportation.Total, Communucation1$Communucation.Total))
        
        colnames(merge_df) <<- c("Total.Food", "Total.NFIs", "Total.Rent", "Total.Utilities", "Total.Education", "Total.Health", "Total.Transportation", "Total.Communucation")
        merge_df <<- cbind(date3, merge_df)
        merge_df$MEB.Total <- rowSums(merge_df[, c("Total.Food", "Total.NFIs", "Total.Rent", "Total.Utilities", "Total.Education", "Total.Health", "Total.Transportation", "Total.Communucation")])
        merge_df1 <<- cbind(merge_df, enf_rate)
        
        
        all_components <- cbind(FOOD, NFI1, rent1, utilities1, EDUCATION1, HEALTH1, Transportation1, Communucation1)
        
        wb <- createWorkbook()
        addWorksheet(wb, "Veri1")
        writeData(wb, sheet = "Veri1", x = merge_df)
        
        addWorksheet(wb, "Veri2")
        writeData(wb, sheet = "Veri2", x = all_components)
        
        addWorksheet(wb, "Veri3")
        writeData(wb, sheet = "Veri3", x = meb_new)
        
        addWorksheet(wb, "Veri4")
        writeData(wb, sheet = "Veri4", x = meb_multiplier)
        
        saveWorkbook(wb, file = "MEB_Calculation.xlsx", overwrite = TRUE)
        
        output$veriCikisi1 <- renderTable({
          merge_df$date <- as.character(merge_df$date)
          merge_df
        })
        
        output$veriCikisi2 <- renderTable({
          FOOD$date <- as.character(FOOD$date)
          FOOD
        })
        
        output$veriCikisi3 <- renderTable({
          NFI$date <- as.character(NFI$date)
          NFI
        })
        
        output$veriCikisi4 <- renderTable({
          shelter$date <- as.character(shelter$date)
          shelter
        })
        
        output$veriCikisi5 <- renderTable({
          other_items$date <- as.character(other_items$date)
          other_items
        })
        
        output$MEB_Calculated <- downloadHandler(
          filename = function() {
            "MEB_Calculation.xlsx"
          },
          content = function(file) {
            file.copy("MEB_Calculation.xlsx", file)
          }
        )
        
        observeEvent(input$dateSlider, {
          selectedDateRange <- input$dateSlider
          
          observeEvent(input$submitButton, {
            selectedVariables <- input$selectedVar
            selectedVariables1 <- input$selectedVar1
            filteredData <- merge_df1[, c("date", selectedVariables, selectedVariables1)]
            filteredData <- filteredData[filteredData$date >= selectedDateRange[1] & filteredData$date <= selectedDateRange[2], ]
            meltedData <- tidyr::gather(filteredData, key = "variable", value = "value", -date)
            
            meltedData$variable <- as.character(meltedData$variable)
            meltedData$value <- as.numeric(meltedData$value)
            
            output$myPlot <- renderPlot({
              ggplot(meltedData, aes(x = date)) +
                geom_bar(data = filter(meltedData, variable %in% selectedVariables), stat = "identity", position = "dodge", aes(y = value, fill = variable)) +
                geom_line(data = filter(meltedData, variable %in% selectedVariables1), aes(y = value * 15000, group = variable, color = variable), linewidth = 2) +
                labs(title = "Monthly amount of MEB components and Annual Inflation Rates",
                     x = "Date", y = "MEB Component Value",
                     fill = "MEB Components", color ="CPI") +
                scale_y_continuous(sec.axis = sec_axis(~./15000, name = "Inflation Rates")) +
                scale_fill_manual(values = c("orangered", "forestgreen", "firebrick", "olivedrab4", "lightseagreen", "bisque2", "linen", "plum", "royalblue")) +
                scale_color_manual(values = c("navyblue", "ivory3", "goldenrod", "salmon4", "darkorchid", "burlywood3", "rosybrown", "cornflowerblue")) + 
                theme_stata() + theme(legend.position = "top",
                                      legend.key = element_rect(color = "grey50"),
                                      legend.key.width = unit(0.9, "cm"),
                                      legend.key.height = unit(0.75, "cm")
                )
            })
            
            output$myPlot1 <- renderPlot({
              ggplot(meltedData, aes(x = date)) +
                geom_bar(data = filter(meltedData, variable %in% selectedVariables), stat = "identity", position = "dodge", aes(y = value, fill = variable)) +
                labs(title = "Monthly amount of MEB components",
                     x = "Date", y = "MEB Component Value",
                     fill = "MEB Components") +
                scale_fill_manual(values = c("orangered", "forestgreen", "firebrick", "olivedrab4", "lightseagreen", "bisque2", "linen", "plum", "royalblue")) + 
                theme_stata() + theme(legend.position = "top",
                                      legend.key = element_rect(color = "grey50"),
                                      legend.key.width = unit(0.9, "cm"),
                                      legend.key.height = unit(0.75, "cm")
                )
            })
            
            output$myPlot2 <- renderPlot({
              ggplot(meltedData, aes(x = date)) +
                geom_line(data = filter(meltedData, variable %in% selectedVariables1), aes(y = value, group = variable, color = variable), linewidth = 2) +
                labs(title = "Annual Inflation Rate",
                     x = "Date", y = "Inflation Rate",
                     color = "CPI") +
                scale_color_manual(values = c("navyblue", "ivory3", "goldenrod", "salmon4", "darkorchid", "burlywood3", "rosybrown", "cornflowerblue")) + 
                theme_stata() + theme(legend.position = "top",
                                      legend.key = element_rect(color = "grey50"),
                                      legend.key.width = unit(0.9, "cm"),
                                      legend.key.height = unit(0.75, "cm")
                )
            })
          })
        })
      })
    }
  })
  observeEvent(input$info_dropdown, {
    selected_option <- input$info_dropdown
    
    if (selected_option == "Ineligible") {
      veri1 <- reactive({
        req(input$dosyaYukleme1)
        dosya <- input$dosyaYukleme1
        if (is.null(dosya))
          return(NULL)
        veri <- read_xlsx(dosya$datapath, skip = 4)
        return(veri)
      })
      
      date3 <- reactive({
        inf <- veri1()
        colnames(inf)[1:3] <- c("Yıl", "Ay", "Months")
        inf <- head(inf, -5)
        enflasyon2 <- inf[-c(1, 2), ]
        ilk_uc_sutun <- enflasyon2[, 1:3]
        
        secilen_sutunlar <- c("01111", "01113", "01146", "01145", "01148", "01151", "01172", "01176", 
                              "01181", "01193", "01230", "09740", "06111", "06231", "04110", "04411", 
                              "04510", "04522", "05611", "13120", "08320", "07321")
        
        enflasyon_fin <- enflasyon2[, secilen_sutunlar]
        
        Tuik <- cbind(ilk_uc_sutun, enflasyon_fin)
        
        
        baslangic_tarihi <- as.Date("2005-01-01")
        
        date <- seq(baslangic_tarihi, by = "1 month", length.out = nrow(Tuik))
        
        return(date)
      })
      
      output$dateSlider <- renderUI({
        sliderInput("dateSlider", "Select Date Range", 
                    min = as.Date("2025-12-01"), 
                    max = max(date3()), 
                    value = c(as.Date("2025-12-01"), max(date3())),
                    step = 30,
                    timeFormat = "%Y-%m-%d")
      })
      
      observeEvent(input$islemButonu, {
        inf <- veri1()
        colnames(inf)[1:3] <- c("Yıl", "Ay", "Months")
        inf <- head(inf, -5)
        enflasyon2 <- inf[-c(1, 2), ]
        ilk_uc_sutun <- enflasyon2[, 1:3]
        
        secilen_sutunlar <- c("01111", "01113", "01146", "01145", "01148", "01151", "01172", "01176", 
                              "01181", "01193", "01230", "09740", "06111", "06231", "04110", "04411", 
                              "04510", "04522", "05611", "13120", "08320", "07321")      
        enflasyon_fin <- enflasyon2[, secilen_sutunlar]
        
        Tuik <<- cbind(ilk_uc_sutun, enflasyon_fin)
        
        
        baslangic_tarihi <- as.Date("2005-01-01")
        
        
        date <- seq(baslangic_tarihi, by = "1 month", length.out = nrow(Tuik))
        Date <- as.data.frame(date)
        date3 <- tail(Date, -251)
        Date1 <- tail(Date, -1)
        enf_sutun <- c("0", "011", "04", "06", "07", "08", "10", "13")
        enf_index <- enflasyon2[, enf_sutun]
        colnames(enf_index) <- c("Inflation.Rate", "Food.Inflation.Rate", "Shelter.Inflation.Rate", "Health.Inflation.Rate",
                                 "Transportation.Inflation.Rate", "Communucation.Inflation.Rate", "Education.Inflation.Rate", "NFIs")
        
        enf_index <- enf_index %>% mutate_all(as.numeric)
        enf_rate <- data.frame()
        for (i in 1:nrow(enf_index)) {
          for (j in 1:ncol(enf_index)) {
            if (i > 12 && enf_index[i, j] != 0) {
              enf_rate[i, j] <- (enf_index[i, j] / enf_index[i - 12, j] -1)
            } else {
              enf_rate[i, j] <- NA
            }
          }
        }
        enf_rate <- tail(enf_rate, -251)
        colnames(enf_rate) <- c("Inflation.Rate", "Food.Inflation.Rate", "Shelter.Inflation.Rate", "Health.Inflation.Rate",
                                "Transportation.Inflation.Rate", "Communucation.Inflation.Rate", "Education.Inflation.Rate", "NFIs")
        
        # GIDA VE ALKOLSÜZ İÇECEKLER
        pirinç <- as.numeric(Tuik$"01111")
        bulgur <- as.numeric(Tuik$"01111")
        ekmek <- as.numeric(Tuik$"01113")
        yogurt <- as.numeric(Tuik$"01146")
        peynir <- as.numeric(Tuik$"01145")
        yumurta <- as.numeric(Tuik$"01148")
        yag <- as.numeric(Tuik$"01151") 
        domates <- as.numeric(Tuik$"01172")
        salatalık <- as.numeric(Tuik$"01172")
        fasulye <- as.numeric(Tuik$"01176") 
        şeker <- as.numeric(Tuik$"01181")
        tuz <- as.numeric(Tuik$"01193")
        çay <- as.numeric(Tuik$"01230")
        
        # EĞİTİM VE KIRTASİYE
        defter <- as.numeric(Tuik$"09740")
        kalem <- as.numeric(Tuik$"09740")
        kırtasiye <- as.numeric(Tuik$"09740")
        
        # SAĞLIK VE KONUT
        ilaç <- as.numeric(Tuik$"06111")
        doktor <- as.numeric(Tuik$"06231")
        kira <- as.numeric(Tuik$"04110")
        su <- as.numeric(Tuik$"04411")
        elektrik <- as.numeric(Tuik$"04510")
        gaz <- as.numeric(Tuik$"04522")
        
        # TEMİZLİK VE KİŞİSEL BAKIM
        çamasır_det <- as.numeric(Tuik$"05611")
        bulasık_det <- as.numeric(Tuik$"05611")
        dezenfektan <- as.numeric(Tuik$"05611")
        
        # Kişisel bakım ürünleri genel olarak 12132 kodu altında toplanmıştır
        tıraş <- as.numeric(Tuik$"13120")
        agız_bakım <- as.numeric(Tuik$"13120")
        sabun <- as.numeric(Tuik$"13120")
        şampuan <- as.numeric(Tuik$"13120")
        tuvalet_kagıdı <- as.numeric(Tuik$"13120")
        bebek_bezi <- as.numeric(Tuik$"13120")
        hijyenik_ped <- as.numeric(Tuik$"13120")
        
        # ULAŞTIRMA VE HABERLEŞME
        telefon <- as.numeric(Tuik$"08320")
        otobüs <- as.numeric(Tuik$"07321")
        
        
        meb <- as.data.frame(cbind(pirinç, bulgur, ekmek, yogurt, peynir, yumurta, yag, domates, salatalık, fasulye, şeker,tuz, çay, defter, kalem, kırtasiye, ilaç, doktor, kira, su,
                                   elektrik, gaz, çamasır_det, bulasık_det, dezenfektan, tıraş, agız_bakım, sabun, şampuan, tuvalet_kagıdı, bebek_bezi, hijyenik_ped, telefon,otobüs))
        meb <- cbind(Date, meb)
        meb_new <- tail(meb, -251)
        meb_multiplier <- data.frame()
        
        for (i in 1:nrow(meb)) {
          for (j in 1:ncol(meb)) {
            if (i == 1) {
              meb_multiplier[i, j] <- NA
            } else if (is.numeric(meb[, j])) {
              meb_multiplier[i, j] <- meb[i, j] / meb[i - 1, j]
            }
          }
        }
        
        meb_multiplier <- tail(meb_multiplier, -1)
        meb_multiplier <- meb_multiplier[, -1]
        
        meb_multiplier <- cbind(Date1, meb_multiplier)
        meb_multiplier <- tail(meb_multiplier, -251)
        
        
        colnames(meb_multiplier) <- colnames(price_ssn)
        
        carpim_sonuclari <- meb_multiplier
        
        
        for (i in 1:ncol(meb_multiplier)) {
          if (is.numeric(meb_multiplier[, i])) {
            if (nrow(meb_multiplier) == 1) {
              carpim_sonuclari[1, i] <- meb_multiplier[1, i]
            } else {
              for (j in 2:nrow(meb_multiplier)) {
                carpim_sonuclari[j, i] <- meb_multiplier[j, i] * carpim_sonuclari[j - 1, i]
              }
            }
          }
        }
        
        price_ssn <- price_ssn[, -1]
        carpim_sonuclari <- carpim_sonuclari[, -1]
        result <- data.frame()
        
        for (i in 1:nrow(carpim_sonuclari)) {
          row <- carpim_sonuclari[i, ] * price_ssn
          result <- rbind(result, row)
        }
        
        
        result <- rbind(price_ssn, result)
        meb_cal <- cbind(date3, result)
        
        meb_multi <- data.frame(
          pirinç = 15,
          bulgur = 7.5,
          ekmek = 37.5,
          yogurt = 7.5,
          peynir = 7.5,
          yumurta = 150,
          yag = 3.75,
          domates = 4.5,
          salatalık = 4.5,
          fasulye = 7.5,
          şeker = 7.5,
          tuz = 0.75,
          çay = 0.75,
          defter = 2,
          kalem = 2,
          kırtasiye = 2,
          ilaç = 3,
          doktor = 3,
          kira = 1,
          su = 15,
          elektrik = 208.333,
          gaz = 1,
          çamasır_det = 1.5,
          bulaşık_det = 0.75,
          dezenfektan = 0.5,
          tıraş = 2,
          ağız_bakım = 1,
          sabun = 1.5,
          şampuan = 0.65,
          tuvalet_kâğıdı = 12,
          bebek_bezi = 150,
          hijyenik_ped = 30,
          telefon = 1,
          otobüs = 32
        )
        
        meb_cal <- meb_cal[, -1]
        
        colnames(meb_multi) <- colnames(meb_cal)
        
        meb_final <- data.frame()
        for (i in seq_len(nrow(meb_cal))) {
          row <- meb_cal[i, ]
          num_cols <- sapply(row, is.numeric)
          row[num_cols] <- row[num_cols] * meb_multi[num_cols]
          meb_final <- rbind(meb_final, row)
        }
        
        meb_cal <- cbind(date3, meb_final)
        
        gıda <- c("date", "Rice", "Bulgur", "Bread", "Yogurt", "White cheese",
                  "Egg", "Sunflower oil", "Tomatoes", "Cucumber", "Dry beans", "Sugar",
                  "Salt", "Tea")
        non_food <- c("date", "Laundry detergent", "Dishwashing liquid", "Disinfectant", "Shaving articles",
                      "Dental hygiene", "Soap", "Haircare", "Toilet paper", "Diaper", "Hygiene Pad for Women")
        edu <- c("date", "Notebook", "Pencil", "Other stationary")
        hp <- c("date", "Medicine", "Specialist")
        transport <- c("date", "Public transportation")
        commun <- c("date", "Mobile service package")
        kira <- c("date", "Rent")
        fatura <- c("date", "Water", "Electricity", "Gas canister (12 L)")
        barinma <- c("date", "Rent", "Water", "Electricity", "Gas canister (12 L)")
        diger <- c("date", "Notebook", "Pencil", "Other stationary", "Medicine", "Specialist", "Public transportation", "Mobile service package")
        
        FOOD <- meb_cal[,gıda]
        NFI <- meb_cal[, non_food]
        EDUCATION <- meb_cal[, edu]
        HEALTH <- meb_cal[, hp]
        Transportation <- meb_cal[, transport]
        Communucation <- meb_cal[, commun]
        rent <- meb_cal[, kira]
        utilities <- meb_cal[, fatura]
        shelter <- meb_cal[, barinma]
        other_items <- meb_cal[, diger]
        
        FOOD$Food.Total <- rowSums(FOOD[, c("Rice", "Bulgur", "Bread", "Yogurt", "White cheese", "Egg", "Sunflower oil", "Tomatoes", "Cucumber", "Dry beans", "Sugar", "Salt", "Tea")])
        NFI$NFI.Total <- rowSums(NFI[, c("Laundry detergent", "Dishwashing liquid", "Disinfectant", "Shaving articles",
                                         "Dental hygiene", "Soap", "Haircare", "Toilet paper", "Diaper", "Hygiene Pad for Women")])
        EDUCATION$Education.Total <- rowSums(EDUCATION[, c("Notebook", "Pencil", "Other stationary")])
        HEALTH$Health.Total <- rowSums(HEALTH[, c("Medicine", "Specialist")])
        Transportation$Transportation.Total <- Transportation$`Public transportation`
        Communucation$Communucation.Total <- Communucation$`Mobile service package`
        rent$Rent.Total <- rent$Rent
        utilities$Utilities.Total <- rowSums(utilities[, c("Water", "Electricity", "Gas canister (12 L)")])
        shelter$Total <- rowSums(shelter[, c("Rent", "Water", "Electricity", "Gas canister (12 L)")])
        other_items$Total <- rowSums(other_items[, c("Notebook", "Pencil", "Other stationary", "Medicine", "Specialist", "Public transportation", "Mobile service package")])
        
        
        NFI1 <- NFI[, -1]
        EDUCATION1 <- EDUCATION[, -1]
        HEALTH1 <- HEALTH[, -1]
        Transportation1 <- Transportation[, -1]
        Communucation1 <-Communucation[, -1]
        rent1 <-rent[, -1]
        utilities1 <-utilities[, -1]
        
        
        merge_df <<- as.data.frame(cbind(FOOD$Food.Total, NFI1$NFI.Total, rent1$Rent.Total, utilities1$Utilities.Total, EDUCATION1$Education.Total, 
                                         HEALTH1$Health.Total, Transportation1$Transportation.Total, Communucation1$Communucation.Total))
        
        colnames(merge_df) <<- c("Total.Food", "Total.NFIs", "Total.Rent", "Total.Utilities", "Total.Education", "Total.Health", "Total.Transportation", "Total.Communucation")
        merge_df <<- cbind(date3, merge_df)
        merge_df$MEB.Total <- rowSums(merge_df[, c("Total.Food", "Total.NFIs", "Total.Rent", "Total.Utilities", "Total.Education", "Total.Health", "Total.Transportation", "Total.Communucation")])
        merge_df1 <<- cbind(merge_df, enf_rate)
        
        
        all_components <- cbind(FOOD, NFI1, rent1, utilities1, EDUCATION1, HEALTH1, Transportation1, Communucation1)
        
        wb <- createWorkbook()
        addWorksheet(wb, "Veri1")
        writeData(wb, sheet = "Veri1", x = merge_df)
        
        addWorksheet(wb, "Veri2")
        writeData(wb, sheet = "Veri2", x = all_components)
        
        addWorksheet(wb, "Veri3")
        writeData(wb, sheet = "Veri3", x = meb_new)
        
        addWorksheet(wb, "Veri4")
        writeData(wb, sheet = "Veri4", x = meb_multiplier)
        
        saveWorkbook(wb, file = "MEB_Calculation.xlsx", overwrite = TRUE)
        
        output$veriCikisi1 <- renderTable({
          merge_df$date <- as.character(merge_df$date)
          merge_df
        })
        
        output$veriCikisi2 <- renderTable({
          FOOD$date <- as.character(FOOD$date)
          FOOD
        })
        
        output$veriCikisi3 <- renderTable({
          NFI$date <- as.character(NFI$date)
          NFI
        })
        
        output$veriCikisi4 <- renderTable({
          shelter$date <- as.character(shelter$date)
          shelter
        })
        
        output$veriCikisi5 <- renderTable({
          other_items$date <- as.character(other_items$date)
          other_items
        })
        
        output$MEB_Calculated <- downloadHandler(
          filename = function() {
            "MEB_Calculation.xlsx"
          },
          content = function(file) {
            file.copy("MEB_Calculation.xlsx", file)
          }
        )
        
        observeEvent(input$dateSlider, {
          selectedDateRange <- input$dateSlider
          
          observeEvent(input$submitButton, {
            selectedVariables <- input$selectedVar
            selectedVariables1 <- input$selectedVar1
            filteredData <- merge_df1[, c("date", selectedVariables, selectedVariables1)]
            filteredData <- filteredData[filteredData$date >= selectedDateRange[1] & filteredData$date <= selectedDateRange[2], ]
            meltedData <- tidyr::gather(filteredData, key = "variable", value = "value", -date)
            
            meltedData$variable <- as.character(meltedData$variable)
            meltedData$value <- as.numeric(meltedData$value)
            
            output$myPlot <- renderPlot({
              ggplot(meltedData, aes(x = date)) +
                geom_bar(data = filter(meltedData, variable %in% selectedVariables), stat = "identity", position = "dodge", aes(y = value, fill = variable)) +
                geom_line(data = filter(meltedData, variable %in% selectedVariables1), aes(y = value * 15000, group = variable, color = variable), linewidth = 2) +
                labs(title = "Monthly amount of MEB components and Annual Inflation Rates",
                     x = "Date", y = "MEB Component Value",
                     fill = "MEB Components", color ="CPI") +
                scale_y_continuous(sec.axis = sec_axis(~./15000, name = "Inflation Rates")) +
                scale_fill_manual(values = c("orangered", "forestgreen", "firebrick", "olivedrab4", "lightseagreen", "bisque2", "linen", "plum", "royalblue")) +
                scale_color_manual(values = c("navyblue", "ivory3", "goldenrod", "salmon4", "darkorchid", "burlywood3", "rosybrown", "cornflowerblue")) + 
                theme_stata() + theme(legend.position = "top",
                                      legend.key = element_rect(color = "grey50"),
                                      legend.key.width = unit(0.9, "cm"),
                                      legend.key.height = unit(0.75, "cm")
                )
            })
            
            output$myPlot1 <- renderPlot({
              ggplot(meltedData, aes(x = date)) +
                geom_bar(data = filter(meltedData, variable %in% selectedVariables), stat = "identity", position = "dodge", aes(y = value, fill = variable)) +
                labs(title = "Monthly amount of MEB components",
                     x = "Date", y = "MEB Component Value",
                     fill = "MEB Components") +
                scale_fill_manual(values = c("orangered", "forestgreen", "firebrick", "olivedrab4", "lightseagreen", "bisque2", "linen", "plum", "royalblue")) + 
                theme_stata() + theme(legend.position = "top",
                                      legend.key = element_rect(color = "grey50"),
                                      legend.key.width = unit(0.9, "cm"),
                                      legend.key.height = unit(0.75, "cm")
                )
            })
            
            output$myPlot2 <- renderPlot({
              ggplot(meltedData, aes(x = date)) +
                geom_line(data = filter(meltedData, variable %in% selectedVariables1), aes(y = value, group = variable, color = variable), linewidth = 2) +
                labs(title = "Annual Inflation Rate",
                     x = "Date", y = "Inflation Rate",
                     color = "CPI") +
                scale_color_manual(values = c("navyblue", "ivory3", "goldenrod", "salmon4", "darkorchid", "burlywood3", "rosybrown", "cornflowerblue")) + 
                theme_stata() + theme(legend.position = "top",
                                      legend.key = element_rect(color = "grey50"),
                                      legend.key.width = unit(0.9, "cm"),
                                      legend.key.height = unit(0.75, "cm")
                )
            })
          })
        })
      })
    }
  })
  observeEvent(input$info_dropdown, {
    selected_option <- input$info_dropdown
    
    if (selected_option == "Turkish") {
      veri1 <- reactive({
        req(input$dosyaYukleme1)
        dosya <- input$dosyaYukleme1
        if (is.null(dosya))
          return(NULL)
        veri <- read_xlsx(dosya$datapath, skip = 4)
        return(veri)
      })
      
      date3 <- reactive({
        inf <- veri1()
        colnames(inf)[1:3] <- c("Yıl", "Ay", "Months")
        inf <- head(inf, -5)
        enflasyon2 <- inf[-c(1, 2), ]
        ilk_uc_sutun <- enflasyon2[, 1:3]
        
        secilen_sutunlar <- c("01111", "01113", "01146", "01145", "01148", "01151", "01172", "01176", 
                              "01181", "01193", "01230", "09740", "06111", "06231", "04110", "04411", 
                              "04510", "04522", "05611", "13120", "08320", "07321")
        
        enflasyon_fin <- enflasyon2[, secilen_sutunlar]
        
        Tuik <- cbind(ilk_uc_sutun, enflasyon_fin)
        
        
        baslangic_tarihi <- as.Date("2005-01-01")
        
        date <- seq(baslangic_tarihi, by = "1 month", length.out = nrow(Tuik))
        
        return(date)
      })
      
      output$dateSlider <- renderUI({
        sliderInput("dateSlider", "Select Date Range", 
                    min = as.Date("2025-12-01"), 
                    max = max(date3()), 
                    value = c(as.Date("2025-12-01"), max(date3())),
                    step = 30,
                    timeFormat = "%Y-%m-%d")
      })
      
      observeEvent(input$islemButonu, {
        inf <- veri1()
        colnames(inf)[1:3] <- c("Yıl", "Ay", "Months")
        inf <- head(inf, -5)
        enflasyon2 <- inf[-c(1, 2), ]
        ilk_uc_sutun <- enflasyon2[, 1:3]
        
        secilen_sutunlar <- c("01111", "01113", "01146", "01145", "01148", "01151", "01172", "01176", 
                              "01181", "01193", "01230", "09740", "06111", "06231", "04110", "04411", 
                              "04510", "04522", "05611", "13120", "08320", "07321")      
        enflasyon_fin <- enflasyon2[, secilen_sutunlar]
        
        Tuik <<- cbind(ilk_uc_sutun, enflasyon_fin)
        
        
        baslangic_tarihi <- as.Date("2005-01-01")
        
        
        date <- seq(baslangic_tarihi, by = "1 month", length.out = nrow(Tuik))
        Date <- as.data.frame(date)
        date3 <- tail(Date, -251)
        Date1 <- tail(Date, -1)
        enf_sutun <- c("0", "011", "04", "06", "07", "08", "10", "13")
        enf_index <- enflasyon2[, enf_sutun]
        colnames(enf_index) <- c("Inflation.Rate", "Food.Inflation.Rate", "Shelter.Inflation.Rate", "Health.Inflation.Rate",
                                 "Transportation.Inflation.Rate", "Communucation.Inflation.Rate", "Education.Inflation.Rate", "NFIs")
        
        enf_index <- enf_index %>% mutate_all(as.numeric)
        enf_rate <- data.frame()
        for (i in 1:nrow(enf_index)) {
          for (j in 1:ncol(enf_index)) {
            if (i > 12 && enf_index[i, j] != 0) {
              enf_rate[i, j] <- (enf_index[i, j] / enf_index[i - 12, j] -1)
            } else {
              enf_rate[i, j] <- NA
            }
          }
        }
        enf_rate <- tail(enf_rate, -251)
        colnames(enf_rate) <- c("Inflation.Rate", "Food.Inflation.Rate", "Shelter.Inflation.Rate", "Health.Inflation.Rate",
                                "Transportation.Inflation.Rate", "Communucation.Inflation.Rate", "Education.Inflation.Rate", "NFIs")
        
        # GIDA VE ALKOLSÜZ İÇECEKLER
        pirinç <- as.numeric(Tuik$"01111")
        bulgur <- as.numeric(Tuik$"01111")
        ekmek <- as.numeric(Tuik$"01113")
        yogurt <- as.numeric(Tuik$"01146")
        peynir <- as.numeric(Tuik$"01145")
        yumurta <- as.numeric(Tuik$"01148")
        yag <- as.numeric(Tuik$"01151") 
        domates <- as.numeric(Tuik$"01172")
        salatalık <- as.numeric(Tuik$"01172")
        fasulye <- as.numeric(Tuik$"01176") 
        şeker <- as.numeric(Tuik$"01181")
        tuz <- as.numeric(Tuik$"01193")
        çay <- as.numeric(Tuik$"01230")
        
        # EĞİTİM VE KIRTASİYE
        defter <- as.numeric(Tuik$"09740")
        kalem <- as.numeric(Tuik$"09740")
        kırtasiye <- as.numeric(Tuik$"09740")
        
        # SAĞLIK VE KONUT
        ilaç <- as.numeric(Tuik$"06111")
        doktor <- as.numeric(Tuik$"06231")
        kira <- as.numeric(Tuik$"04110")
        su <- as.numeric(Tuik$"04411")
        elektrik <- as.numeric(Tuik$"04510")
        gaz <- as.numeric(Tuik$"04522")
        
        # TEMİZLİK VE KİŞİSEL BAKIM
        çamasır_det <- as.numeric(Tuik$"05611")
        bulasık_det <- as.numeric(Tuik$"05611")
        dezenfektan <- as.numeric(Tuik$"05611")
        
        # Kişisel bakım ürünleri genel olarak 12132 kodu altında toplanmıştır
        tıraş <- as.numeric(Tuik$"13120")
        agız_bakım <- as.numeric(Tuik$"13120")
        sabun <- as.numeric(Tuik$"13120")
        şampuan <- as.numeric(Tuik$"13120")
        tuvalet_kagıdı <- as.numeric(Tuik$"13120")
        bebek_bezi <- as.numeric(Tuik$"13120")
        hijyenik_ped <- as.numeric(Tuik$"13120")
        
        # ULAŞTIRMA VE HABERLEŞME
        telefon <- as.numeric(Tuik$"08320")
        otobüs <- as.numeric(Tuik$"07321")
        
        
        meb <- as.data.frame(cbind(pirinç, bulgur, ekmek, yogurt, peynir, yumurta, yag, domates, salatalık, fasulye, şeker,tuz, çay, defter, kalem, kırtasiye, ilaç, doktor, kira, su,
                                   elektrik, gaz, çamasır_det, bulasık_det, dezenfektan, tıraş, agız_bakım, sabun, şampuan, tuvalet_kagıdı, bebek_bezi, hijyenik_ped, telefon,otobüs))
        meb <- cbind(Date, meb)
        meb_new <- tail(meb, -251)
        meb_multiplier <- data.frame()
        
        for (i in 1:nrow(meb)) {
          for (j in 1:ncol(meb)) {
            if (i == 1) {
              meb_multiplier[i, j] <- NA
            } else if (is.numeric(meb[, j])) {
              meb_multiplier[i, j] <- meb[i, j] / meb[i - 1, j]
            }
          }
        }
        
        meb_multiplier <- tail(meb_multiplier, -1)
        meb_multiplier <- meb_multiplier[, -1]
        
        meb_multiplier <- cbind(Date1, meb_multiplier)
        meb_multiplier <- tail(meb_multiplier, -251)
        
        
        colnames(meb_multiplier) <- colnames(price_turk)
        
        carpim_sonuclari <- meb_multiplier
        
        
        for (i in 1:ncol(meb_multiplier)) {
          if (is.numeric(meb_multiplier[, i])) {
            if (nrow(meb_multiplier) == 1) {
              carpim_sonuclari[1, i] <- meb_multiplier[1, i]
            } else {
              for (j in 2:nrow(meb_multiplier)) {
                carpim_sonuclari[j, i] <- meb_multiplier[j, i] * carpim_sonuclari[j - 1, i]
              }
            }
          }
        }
        
        price_turk <- price_turk[, -1]
        carpim_sonuclari <- carpim_sonuclari[, -1]
        result <- data.frame()
        
        for (i in 1:nrow(carpim_sonuclari)) {
          row <- carpim_sonuclari[i, ] * price_turk
          result <- rbind(result, row)
        }
        
        
        result <- rbind(price_turk, result)
        meb_cal <- cbind(date3, result)
        
        meb_multi <- data.frame(
          pirinç = 15,
          bulgur = 7.5,
          ekmek = 37.5,
          yogurt = 7.5,
          peynir = 7.5,
          yumurta = 150,
          yag = 3.75,
          domates = 4.5,
          salatalık = 4.5,
          fasulye = 7.5,
          şeker = 7.5,
          tuz = 0.75,
          çay = 0.75,
          defter = 2,
          kalem = 2,
          kırtasiye = 2,
          ilaç = 3,
          doktor = 3,
          kira = 1,
          su = 18,
          elektrik = 208.333,
          gaz = 1,
          çamasır_det = 1,
          bulaşık_det = 0.65,
          dezenfektan = 0.6,
          tıraş = 2,
          ağız_bakım = 2,
          sabun = 1.2,
          şampuan = 0.65,
          tuvalet_kâğıdı = 10,
          bebek_bezi = 90,
          hijyenik_ped = 30,
          telefon = 1,
          otobüs = 32
        )
        
        meb_cal <- meb_cal[, -1]
        
        colnames(meb_multi) <- colnames(meb_cal)
        
        meb_final <- data.frame()
        for (i in seq_len(nrow(meb_cal))) {
          row <- meb_cal[i, ]
          num_cols <- sapply(row, is.numeric)
          row[num_cols] <- row[num_cols] * meb_multi[num_cols]
          meb_final <- rbind(meb_final, row)
        }
        
        meb_cal <- cbind(date3, meb_final)
        
        gıda <- c("date", "Rice", "Bulgur", "Bread", "Yogurt", "White cheese",
                  "Egg", "Sunflower oil", "Tomatoes", "Cucumber", "Dry beans", "Sugar",
                  "Salt", "Tea")
        non_food <- c("date", "Laundry detergent", "Dishwashing liquid", "Disinfectant", "Shaving articles",
                      "Dental hygiene", "Soap", "Haircare", "Toilet paper", "Diaper", "Hygiene Pad for Women")
        edu <- c("date", "Notebook", "Pencil", "Other stationary")
        hp <- c("date", "Medicine", "Specialist")
        transport <- c("date", "Public transportation")
        commun <- c("date", "Mobile service package")
        kira <- c("date", "Rent")
        fatura <- c("date", "Water", "Electricity", "Gas canister (12 L)")
        barinma <- c("date", "Rent", "Water", "Electricity", "Gas canister (12 L)")
        diger <- c("date", "Notebook", "Pencil", "Other stationary", "Medicine", "Specialist", "Public transportation", "Mobile service package")
        
        FOOD <- meb_cal[,gıda]
        NFI <- meb_cal[, non_food]
        EDUCATION <- meb_cal[, edu]
        HEALTH <- meb_cal[, hp]
        Transportation <- meb_cal[, transport]
        Communucation <- meb_cal[, commun]
        rent <- meb_cal[, kira]
        utilities <- meb_cal[, fatura]
        shelter <- meb_cal[, barinma]
        other_items <- meb_cal[, diger]
        
        FOOD$Food.Total <- rowSums(FOOD[, c("Rice", "Bulgur", "Bread", "Yogurt", "White cheese", "Egg", "Sunflower oil", "Tomatoes", "Cucumber", "Dry beans", "Sugar", "Salt", "Tea")])
        NFI$NFI.Total <- rowSums(NFI[, c("Laundry detergent", "Dishwashing liquid", "Disinfectant", "Shaving articles",
                                         "Dental hygiene", "Soap", "Haircare", "Toilet paper", "Diaper", "Hygiene Pad for Women")])
        EDUCATION$Education.Total <- rowSums(EDUCATION[, c("Notebook", "Pencil", "Other stationary")])
        HEALTH$Health.Total <- rowSums(HEALTH[, c("Medicine", "Specialist")])
        Transportation$Transportation.Total <- Transportation$`Public transportation`
        Communucation$Communucation.Total <- Communucation$`Mobile service package`
        rent$Rent.Total <- rent$Rent
        utilities$Utilities.Total <- rowSums(utilities[, c("Water", "Electricity", "Gas canister (12 L)")])
        shelter$Total <- rowSums(shelter[, c("Rent", "Water", "Electricity", "Gas canister (12 L)")])
        other_items$Total <- rowSums(other_items[, c("Notebook", "Pencil", "Other stationary", "Medicine", "Specialist", "Public transportation", "Mobile service package")])
        
        
        NFI1 <- NFI[, -1]
        EDUCATION1 <- EDUCATION[, -1]
        HEALTH1 <- HEALTH[, -1]
        Transportation1 <- Transportation[, -1]
        Communucation1 <-Communucation[, -1]
        rent1 <-rent[, -1]
        utilities1 <-utilities[, -1]
        
        
        merge_df <<- as.data.frame(cbind(FOOD$Food.Total, NFI1$NFI.Total, rent1$Rent.Total, utilities1$Utilities.Total, EDUCATION1$Education.Total, 
                                         HEALTH1$Health.Total, Transportation1$Transportation.Total, Communucation1$Communucation.Total))
        
        colnames(merge_df) <<- c("Total.Food", "Total.NFIs", "Total.Rent", "Total.Utilities", "Total.Education", "Total.Health", "Total.Transportation", "Total.Communucation")
        merge_df <<- cbind(date3, merge_df)
        merge_df$MEB.Total <- rowSums(merge_df[, c("Total.Food", "Total.NFIs", "Total.Rent", "Total.Utilities", "Total.Education", "Total.Health", "Total.Transportation", "Total.Communucation")])
        merge_df1 <<- cbind(merge_df, enf_rate)
        
        
        all_components <- cbind(FOOD, NFI1, rent1, utilities1, EDUCATION1, HEALTH1, Transportation1, Communucation1)
        
        wb <- createWorkbook()
        addWorksheet(wb, "Veri1")
        writeData(wb, sheet = "Veri1", x = merge_df)
        
        addWorksheet(wb, "Veri2")
        writeData(wb, sheet = "Veri2", x = all_components)
        
        addWorksheet(wb, "Veri3")
        writeData(wb, sheet = "Veri3", x = meb_new)
        
        addWorksheet(wb, "Veri4")
        writeData(wb, sheet = "Veri4", x = meb_multiplier)
        
        saveWorkbook(wb, file = "MEB_Calculation.xlsx", overwrite = TRUE)
        
        output$veriCikisi1 <- renderTable({
          merge_df$date <- as.character(merge_df$date)
          merge_df
        })
        
        output$veriCikisi2 <- renderTable({
          FOOD$date <- as.character(FOOD$date)
          FOOD
        })
        
        output$veriCikisi3 <- renderTable({
          NFI$date <- as.character(NFI$date)
          NFI
        })
        
        output$veriCikisi4 <- renderTable({
          shelter$date <- as.character(shelter$date)
          shelter
        })
        
        output$veriCikisi5 <- renderTable({
          other_items$date <- as.character(other_items$date)
          other_items
        })
        
        output$MEB_Calculated <- downloadHandler(
          filename = function() {
            "MEB_Calculation.xlsx"
          },
          content = function(file) {
            file.copy("MEB_Calculation.xlsx", file)
          }
        )
        
        observeEvent(input$dateSlider, {
          selectedDateRange <- input$dateSlider
          
          observeEvent(input$submitButton, {
            selectedVariables <- input$selectedVar
            selectedVariables1 <- input$selectedVar1
            filteredData <- merge_df1[, c("date", selectedVariables, selectedVariables1)]
            filteredData <- filteredData[filteredData$date >= selectedDateRange[1] & filteredData$date <= selectedDateRange[2], ]
            meltedData <- tidyr::gather(filteredData, key = "variable", value = "value", -date)
            
            meltedData$variable <- as.character(meltedData$variable)
            meltedData$value <- as.numeric(meltedData$value)
            
            output$myPlot <- renderPlot({
              ggplot(meltedData, aes(x = date)) +
                geom_bar(data = filter(meltedData, variable %in% selectedVariables), stat = "identity", position = "dodge", aes(y = value, fill = variable)) +
                geom_line(data = filter(meltedData, variable %in% selectedVariables1), aes(y = value * 15000, group = variable, color = variable), linewidth = 2) +
                labs(title = "Monthly amount of MEB components and Annual Inflation Rates",
                     x = "Date", y = "MEB Component Value",
                     fill = "MEB Components", color ="CPI") +
                scale_y_continuous(sec.axis = sec_axis(~./15000, name = "Inflation Rates")) +
                scale_fill_manual(values = c("orangered", "forestgreen", "firebrick", "olivedrab4", "lightseagreen", "bisque2", "linen", "plum", "royalblue")) +
                scale_color_manual(values = c("navyblue", "ivory3", "goldenrod", "salmon4", "darkorchid", "burlywood3", "rosybrown", "cornflowerblue")) + 
                theme_stata() + theme(legend.position = "top",
                                      legend.key = element_rect(color = "grey50"),
                                      legend.key.width = unit(0.9, "cm"),
                                      legend.key.height = unit(0.75, "cm")
                )
            })
            
            output$myPlot1 <- renderPlot({
              ggplot(meltedData, aes(x = date)) +
                geom_bar(data = filter(meltedData, variable %in% selectedVariables), stat = "identity", position = "dodge", aes(y = value, fill = variable)) +
                labs(title = "Monthly amount of MEB components",
                     x = "Date", y = "MEB Component Value",
                     fill = "MEB Components") +
                scale_fill_manual(values = c("orangered", "forestgreen", "firebrick", "olivedrab4", "lightseagreen", "bisque2", "linen", "plum", "royalblue")) + 
                theme_stata() + theme(legend.position = "top",
                                      legend.key = element_rect(color = "grey50"),
                                      legend.key.width = unit(0.9, "cm"),
                                      legend.key.height = unit(0.75, "cm")
                )
            })
            
            output$myPlot2 <- renderPlot({
              ggplot(meltedData, aes(x = date)) +
                geom_line(data = filter(meltedData, variable %in% selectedVariables1), aes(y = value, group = variable, color = variable), linewidth = 2) +
                labs(title = "Annual Inflation Rate",
                     x = "Date", y = "Inflation Rate",
                     color = "CPI") +
                scale_color_manual(values = c("navyblue", "ivory3", "goldenrod", "salmon4", "darkorchid", "burlywood3", "rosybrown", "cornflowerblue")) + 
                theme_stata() + theme(legend.position = "top",
                                      legend.key = element_rect(color = "grey50"),
                                      legend.key.width = unit(0.9, "cm"),
                                      legend.key.height = unit(0.75, "cm")
                )
            })
          })
        })
      })
    }
  })
}

shinyApp(ui, server, options = list(debug = TRUE))