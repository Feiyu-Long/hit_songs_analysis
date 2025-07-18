##### Load collections #####
library(tidyverse)
library(readxl)  

##### Read file and store in df #####
df <- read_delim("hits_dataset.csv", delim = "\t") |>
  drop_na() |>
  distinct()

##### Bar plot for explicit / non-explicit #####
df |>
  pull(explicit) |>
  factor(labels = c("Non-Explicit", "Explicit")) |>
  plot(main = "Barplot of Explicit Content", ylab = "Count", col = "steelblue")

##### Bar plot for time signature #####
ts_counts <- table(df$time_signature)
barplot(ts_counts,
        main = "Distribution of Time Signatures",
        xlab = "Time Signature",
        ylab = "Count",
        col = "steelblue")

##### Scatterplot and correlation check for a few hip hop artists #####
df_hip_hop <- df |>
  filter(str_detect(name_artists, "Kendrick Lamar|Drake|Travis Scott|Eminem|XXXTENTACION|Lil Wayne")) |>
  select(
    PO = popularity,
    AC = acousticness,
    DA = danceability,
    EN = energy,
    IN= instrumentalness,
    LI = liveness,
    LO = loudness,
    SP = speechiness,
    VA = valence,
    TE = tempo
  ) # select and mutate df

df_hip_hop |> plot() # plot scatterplot matrix

df_hip_hop |> 
  cor() |>
  round(2) # plot correlation matrix to check for strong corr.

model <- lm(EN ~ LO, data = df_hip_hop) # fit model

df_hip_hop |>
  select(LO, EN) |> 
  plot(
    main = "Scatterplot of Energy vs Loudness in Hip-Hop Hits",
    xlab = "Loudness",
    ylab = "Energy",
    col = "red3",
    pch = 16
  ) # plot scatterplot

abline(model, col = "blue", lwd = 2) # fit regression line

##### Clean Duration Column #####
df_clean <- df |>
  mutate(duration_min = duration_ms / 60000) |>
  select(duration_min, popularity) # mutate and simplify df

plot(df_clean$duration_min, df_clean$popularity,
     main = "Scatterplot of Popularity vs Duration in a Hit Song",
     xlab = "Duration (min)",
     ylab = "Popularity",
     col = "red3",
     pch = 20) # plot scatterplot

