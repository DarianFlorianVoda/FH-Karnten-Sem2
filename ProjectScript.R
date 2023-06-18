data = read.csv("responses.csv")
columns = read.csv("columns.csv")

data[141:150]



# Split into DFs
music_dataframe = data.frame(data[1:19])
music_dataframe

movies_dataframe = data.frame(data[20:31])
movies_dataframe

hobbies_dataframe = data.frame(data[32:63])
hobbies_dataframe

phobias_dataframe = data.frame(data[64:73])
phobias_dataframe

health_habits_dataframe = data.frame(data[74:76])
health_habits_dataframe

personality_dataframe = data.frame(data[78:133])
personality_dataframe

spendings_dataframe = data.frame(data[134:140])
spendings_dataframe

demographics_dataframe = data.frame(data[141:150])
demographics_dataframe

# Take a look at general charts

library(ggplot2)

ggplot(music_dataframe, aes(Music)) +
  geom_bar()

