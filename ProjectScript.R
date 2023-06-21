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


  
  
library(shiny)
runExample("01_hello")


runApp("Young_People_Survey", display.mode = "showcase")
