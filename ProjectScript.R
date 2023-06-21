data = read.csv("responses.csv")
columns = read.csv("columns.csv")

data[141:150]



# Split into DFs
music_dataframe = na.omit(data.frame(data[1:19]))
music_dataframe

movies_dataframe = na.omit(data.frame(data[20:31]))
movies_dataframe

hobbies_dataframe = na.omit(data.frame(data[32:63]))
hobbies_dataframe

phobias_dataframe = na.omit(data.frame(data[64:73]))
phobias_dataframe

health_habits_dataframe = na.omit(data.frame(data[74:76]))
health_habits_dataframe

personality_dataframe = na.omit(data.frame(data[78:133]))
personality_dataframe

spendings_dataframe = na.omit(data.frame(data[134:140]))
spendings_dataframe

demographics_dataframe = na.omit(data.frame(data[141:150]))
demographics_dataframe

# Take a look at general charts

library(ggplot2)
library(dplyr)
library(ggrepel)
library(tidyverse)
library(scales)

clrs5 <- c("firebrick4", "firebrick1", "gray70", "blue", "darkblue")

overall_music = music_dataframe %>% 
  group_by(Music) %>%
  summarise(counts = n())

overall_music

music_dataframe

overall_music_plot = ggplot(overall_music, aes(x = Music, y = counts, fill=counts)) +
  geom_bar(stat = "identity", fill="steelblue")+
  geom_text(aes(label= counts), vjust = -0.3)+
  theme(axis.title.x = element_blank(),
        plot.title = element_text(hjust = 0.5))+
  labs(y="Number of respondents")+
  ggtitle("How much people enjoy Music?")+
  geom_text(x=1, y=770, label="1 - not so much", color="red", hjust = 0, size = 5)+
  geom_text(x=1, y=740, label="5 - very much", color = "blue", hjust = 0, size = 5)+
  scale_y_continuous(breaks=round(seq(0, max(overall_music$counts), by=100), 1))

overall_music_plot

# Count each occurrence for a specific type of question
music_data_frame_long = music_dataframe %>%
  count(Music)

music_data_frame_long  

music_data_frame_long =  music_data_frame_long %>% 
  mutate(Percent = round((n/sum(n)),5)*100,
         csum = rev(cumsum(rev(n))), 
         pos = n/2 + lead(csum, 1),
         pos = if_else(is.na(pos), n/2, pos))
  
music_data_frame_long



# Basic piechart
ggplot(music_data_frame_long, aes("", Percent, fill = factor(Music))) + 
  geom_col(width = 1, color = 1) +
  coord_polar("y") +
  scale_fill_brewer(palette = "Pastel1") +
  geom_label_repel(data = music_data_frame_long,
                   aes(y = pos, label = paste0(n, "%")),
                   size = 4.5, nudge_x = 1, show.legend = FALSE) +
  guides(fill = guide_legend(title = "Scale"))+
  theme_void()

# TODO: Create better a heatmap with all these
# TODO: Then tooltip it in ggplotly
# 



## Convert columns to Rows

Question = colnames(music_dataframe)


music_data_frame_long
music_data_frame_long$n


df = data.frame(Question)
df

df = df %>%
  add_column("Strongly_disagree" = NA,
             "Disagree" = NA,
             "Neutral" = NA,
             "Agree" = NA,
             "Strongly_agree" = NA)

df

music_vals = music_data_frame_long$n
music_vals

# Count occurrences of each value in each column
occurrences <- lapply(music_dataframe, table)
values <- lapply(occurrences, function(x) as.vector(unname(x)))
occurrences
values

### INSERT VALUES FOR THE COLUMNS OF EACH QUESTION
df[1, 2:(1 + length(music_vals))] <- c(music_vals, df[2, -(1:(1 + length(music_vals)))])
df[2, 2:(1 + length(values$Slow.songs.or.fast.songs))] <- c(values$Slow.songs.or.fast.songs, df[2, -(1:(1 + length(values$Slow.songs.or.fast.songs)))])
df[3, 2:(1 + length(values$Dance))] <- c(values$Dance, df[2, -(1:(1 + length(values$Dance)))])
df[4, 2:(1 + length(values$Folk))] <- c(values$Folk, df[2, -(1:(1 + length(values$Folk)))])
df[5, 2:(1 + length(values$Country))] <- c(values$Country, df[2, -(1:(1 + length(values$Country)))])
df[6, 2:(1 + length(values$Classical.music))] <- c(values$Classical.music, df[2, -(1:(1 + length(values$Classical.music)))])
df[7, 2:(1 + length(values$Musical))] <- c(values$Musical, df[2, -(1:(1 + length(values$Musical)))])
df[8, 2:(1 + length(values$Pop))] <- c(values$Pop, df[2, -(1:(1 + length(values$Pop)))])
df[9, 2:(1 + length(values$Rock))] <- c(values$Rock, df[2, -(1:(1 + length(values$Rock)))])
df[10, 2:(1 + length(values$Metal.or.Hardrock))] <- c(values$Metal.or.Hardrock, df[2, -(1:(1 + length(values$Metal.or.Hardrock)))])
df[11, 2:(1 + length(values$Punk))] <- c(values$Punk, df[2, -(1:(1 + length(values$Punk)))])
df[12, 2:(1 + length(values$Hiphop..Rap))] <- c(values$Hiphop..Rap, df[2, -(1:(1 + length(values$Hiphop..Rap)))])
df[13, 2:(1 + length(values$Reggae..Ska))] <- c(values$Reggae..Ska, df[2, -(1:(1 + length(values$Reggae..Ska)))])
df[14, 2:(1 + length(values$Swing..Jazz))] <- c(values$Swing..Jazz, df[2, -(1:(1 + length(values$Swing..Jazz)))])
df[15, 2:(1 + length(values$Rock.n.roll))] <- c(values$Rock.n.roll, df[2, -(1:(1 + length(values$Rock.n.roll)))])
df[16, 2:(1 + length(values$Alternative))] <- c(values$Alternative, df[2, -(1:(1 + length(values$Alternative)))])
df[17, 2:(1 + length(values$Latino))] <- c(values$Latino, df[2, -(1:(1 + length(values$Latino)))])
df[18, 2:(1 + length(values$Techno..Trance))] <- c(values$Techno..Trance, df[2, -(1:(1 + length(values$Techno..Trance)))])
df[19, 2:(1 + length(values$Opera))] <- c(values$Opera, df[2, -(1:(1 + length(values$Opera)))])

df

## Stacked percent bar chart

df_long <- tidyr::gather(df, Answer, Occurrence, -Question)
df_long

# Calculate the percentages for each category
df_long <- transform(df_long, Percentage = Occurrence / tapply(Occurrence, Question, sum)[Question] * 100)

# Reorder the levels of the "Answer" variable in the dataframe
df_long$Answer <- factor(df_long$Answer, levels = c("Strongly_agree", "Agree", "Neutral", "Disagree", "Strongly_disagree"))

# Horizontal Stacked Bar Chart
stacked_music_field <- ggplot(df_long, aes(x = Percentage, y = Question, fill = Answer, text = paste("Answer: ", Answer, "<br>Occurrences: ", Occurrence, "<br>Percentage: ", Percentage, "%"))) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = ifelse(Percentage < 3, "", paste0(round(Percentage, 1), "%"))),
            position = position_stack(vjust = 0.5), color = "blue", size = 4) +
  geom_text(aes(label = Occurrence), size = 0, alpha = 0) +  # Add hidden Occurrences
  labs(x = "Percentage", y = "Category") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  ggtitle("How much people enjoy the following Categories?")+
  scale_fill_manual(name = "Scale", values = c("green", "cyan", "grey", "chocolate1", "tomato2"), labels = c("Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"))

# Convert ggplot to a plotly plot
stacked_music_field <- ggplotly(stacked_music_field, tooltip = "text")

# Modify the tooltip text
stacked_music_field$x$data[[1]]$text <- df_long$Answer

# Print the plot
stacked_music_field


library(plotly)


library(shiny)
runExample("01_hello")


runApp("Young_People_Survey", display.mode = "showcase")
