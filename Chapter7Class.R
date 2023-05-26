packages = c("tidyverse", "ggplot2", "maps", "sf", "sp", "classInt", "RColorBrewer")

lapply(packages, install.packages)

lapply(packages, library, character.only = TRUE)

ItalyGeom = map_data("italy")

class(ItalyGeom)

head(ItalyGeom)

ggplot(data = ItalyGeom, mapping = aes(x = long, y = lat, group = group)) + 
  geom_polygon(fill = "white", color = "black")


AT_geom = st_read(dsn = "C:/Users/Dari-Laptop/Desktop/FH Karnten - Master - AppDs/FH-Karnten-Sem2", layer = "austriaNUTS2")


plot(AT_geom)

class(AT_geom)

head(AT_geom$geometry)

View(AT_geom)

ggplot(AT_geom) + geom_sf()

AT_sp = as(AT_geom, "Spatial")

class(AT_geom)

class(AT_sp)

AT_geom2 = fortify(AT_sp)

ggplot(data = AT_geom2, mapping = aes(x = long, y = lat, group = group)) + 
  geom_polygon(fill = "white", color = "black")

head(AT_sp@data)

install.packages("giscoR")
library(giscoR)

nuts3 = gisco_get_nuts(nuts_level = 3)

class(nuts3)

ggplot(nuts3) + geom_sf()

austria_nuts3 = gisco_get_nuts(nuts_level = 3, country = "Austria")

ggplot(austria_nuts3) + geom_sf()

austria_nuts3 = gisco_get_nuts(nuts_level = 3, country = "Austria", resolution = "03")

ggplot(austria_nuts3) + geom_sf()

austria_nuts3 = austria_nuts3 %>% left_join(DEMO, by="NUTS_ID")

austria_nuts3$AERAnew = st_area(austria_nuts3)/1000000

View(austria_nuts3)

austria_nuts3$AERAnew = units::drop_units(austria_nuts3$AERAnew)

austria_nuts3$popDens = austria_nuts3$Total / austria_nuts3$AERAnew

ggplot(austria_nuts3) + geom_sf(aes(fill = popDens))

hist(austria_nuts3$popDens, breaks = 50)

ggplot(austria_nuts3) + geom_sf(aes(fill = cut_number(popDens, n = 5))) + scale_fill_brewer(palette = "Reds")

breaksQ = classIntervals(austria_nuts3$popDens, n = 6, style = "quantile")
breaksJ = classIntervals(austria_nuts3$popDens, n = 6, style = "jenks")
breaksP = classIntervals(austria_nuts3$popDens, n = 6, style = "pretty")
breaksJ$brks

ggplot(austria_nuts3) + geom_sf(aes(fill = cut(popDens, breaksP$brks, include.lowest = T))) + scale_fill_brewer(palette = "Reds")
ggplot(austria_nuts3) + geom_sf(aes(fill = cut(popDens, breaksJ$brks, include.lowest = T))) + scale_fill_brewer(palette = "Reds")

ggplot(austria_nuts3) + geom_sf(aes(fill = cut(popDens, breaksQ$brks, include.lowest = T))) + scale_fill_brewer(palette = "PuBu")


map1 = ggplot(austria_nuts3) + geom_sf(aes(fill = cut(popDens, breaksQ$brks, include.lowest = T)), color = "white") + 
  scale_fill_brewer(palette = "PuBu", labels = c("less then 32", "32.1 - 50.8", "50.9 - 74", "74.1 - 94.9", "95 - 172", "over 172"))

map2 = map1 + labs(title = "Population density in Austria", subtitle = "in 2023", fill = "population density per squared km", caption = "data source: Eurostat")

map2 + theme_minimal() + theme(legend.position = "bottom") + guides(fill = guide_legend(title.position = "top", nrow = 1))

symbolPos = st_centroid(austria_nuts3, of_largest_polygon = T)

plot(symbolPos)

symbolPos = cbind(symbolPos, st_coordinates(symbolPos))

symbol = ggplot(data = austria_nuts3) + geom_sf(fill = "grey60", color = "white")

symbol

symbol + geom_point(data = symbolPos, aes(x = X, y = Y, size = Total), fill = "#EDB86C", shape = 21, alpha = 0.8, color = "grey20")

classIntervals(symbolPos$Total, n = 5, style = "jenks")$brks

myBreaks = c(50000, 100000, 200000, 350000, 600000, 2000000)
myLabels = c("50000", "100000", "200000", "350000", "600000", "2000000")

symbol2 = symbol + geom_point(data = symbolPos, aes(x = X, y = Y, size = Total), fill = "#EDB86C", shape = 21, alpha = 0.8, color = "grey20")


symbol2 + scale_size(range = c(1, 20), name = "total population", breaks = myBreaks, labels = myLabels) + theme_minimal()

install.packages("leaflet")

library(leaflet)

mapWGS84 = st_transform(austria_nuts3, 4326)

leaflet(mapWGS84) %>% addPolygons()

palettCol = colorBin("Reds", domain = austria_nuts3$popDens, bins = breaksQ$brks)

popUp  = paste0("total population: ", austria_nuts3$Total)

leaflet(mapWGS84) %>% addPolygons(color = "white", fillColor = ~palettCol(popDens), popup = popUp, fillOpacity = 0.8) %>% addTiles() %>% 
  addLegend("bottomright", pal = palettCol, values = ~ popDens, title = "population density")
