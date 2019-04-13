#library
library(tidyverse)
library(caret)
library(lubridate)

#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)


ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


# To extract the year of release from the title.
edx <- 
  edx %>% 
  mutate(release_year = as.numeric(str_sub(title,-5,-2)))

validation <- 
  validation %>% 
  mutate(release_year = as.numeric(str_sub(title,-5,-2)))

# Convert timestamp to year-month-date
edx <- 
  edx %>% 
  mutate(date = date(as.POSIXct(timestamp, tz = "UCT" ,origin = origin)))

validation <- 
  validation %>% 
  mutate(date = date(as.POSIXct(timestamp, tz = "UCT" ,origin = origin)))

# Adding the number of days from the initial post.

edx <- edx %>% 
  group_by(movieId) %>% 
  summarize(initial_post_day = min(date)) %>% 
  left_join(edx, by = "movieId") %>% 
  mutate(days = date - initial_post_day) %>% 
  select(-initial_post_day, -timestamp)

validation <- validation %>% 
  group_by(movieId) %>% 
  summarize(initial_post_day = min(date)) %>% 
  left_join(validation, by = "movieId") %>% 
  mutate(days = date - initial_post_day) %>% 
  select(-initial_post_day, -timestamp)

# The edx set's summary is as follows.

edx %>% 
  mutate(movieId = as.factor(movieId),
         userId = as.factor(userId),
         genres = as.factor(genres),
         release_year = as.factor(release_year)) %>% 
         str(edx)

# Rating's summary is as follow.

summary(edx$rating)
hist(edx$rating)

# I write a function that computes the RMSE for vectors of ratings and their corresponding predictors:

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Just the average model

mu <- mean(edx$rating)
mu_hat <- mu
model_1_rmse <- RMSE(validation$rating, mu_hat)


# Movie effects model

movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu)) 

movie_avgs %>% qplot(b_i, geom ="histogram", bins = 100, data = ., color = I("black"))

predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

model_2_rmse <- RMSE(validation$rating, predicted_ratings)


# Movie + User Effects Model
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

user_avgs %>% qplot(b_u, geom ="histogram", bins = 100, data = ., color = I("black"))

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

model_3_rmse <- RMSE(predicted_ratings, validation$rating)


# Movie + User + Genres Effects Model

genres_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_i - b_u)) 

genres_avgs %>% qplot(b_g, geom ="histogram", bins = 100, data = ., color = I("black"))

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_g) %>% 
  pull(pred)

model_4_rmse <- RMSE(validation$rating, predicted_ratings)

# Movie + User + Genres + Days Effects Model

days_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  group_by(days) %>% 
  summarize(b_d = mean(rating - mu - b_i - b_u - b_g))

days_avgs %>% qplot(b_d, geom ="histogram", bins = 100, data = ., color = I("black"))

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  left_join(days_avgs, by='days') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_d) %>%
  pull(pred)

model_5_rmse <- RMSE(predicted_ratings, validation$rating)

# Movie + User + Genres + Days + Release year Effects Model

release_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  left_join(days_avgs, by='days') %>%
  group_by(release_year) %>% 
  summarize(b_y = mean(rating - mu - b_i - b_u - b_g - b_d)) 

release_avgs %>% qplot(b_y, geom ="histogram", bins = 100, data = ., color = I("black"))

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  left_join(days_avgs, by='days') %>%
  left_join(release_avgs, by="release_year") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_d + b_y) %>% 
  pull(pred)

model_6_rmse <- RMSE(predicted_ratings, validation$rating)


# Regularized Movie + User + Genres + Days + Release year Effects Model

lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_g <- edx %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating -b_u - b_i - mu)/(n()+l))
  
  b_d <- edx %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_g, by="genres") %>%    
    group_by(days) %>%
    summarize(b_d = sum(rating - b_g - b_u - b_i - mu)/(n()+l))
  
  b_y <- edx %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_g, by="genres") %>% 
    left_join(b_d, by = "days") %>%
    group_by(release_year) %>%
    summarize(b_y = sum(rating - b_d - b_g - b_u -b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_d, by = "days") %>%    
    left_join(b_y, by = "release_year") %>%
    mutate(pred = mu + b_i + b_u + b_g + b_d + b_y) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})

qplot(lambdas, rmses) 

lambda <- lambdas[which.min(rmses)]
lambda

model_7_rmse <- min(rmses)

# Store all results in a data frame.

rmse_results <- bind_rows(
  data_frame(method = "Just the average Model", 
             RMSE = model_1_rmse),
  data_frame(method = "Movie Effect Model", 
             RMSE = model_2_rmse),
  data_frame(method = "Movie + User Effects Model", 
             RMSE = model_3_rmse),
  data_frame(method = "Movie + User + Genres Effects Model", 
             RMSE = model_4_rmse),
  data_frame(method = "Movie + User + Genres + Days Effects Model", 
             RMSE = model_5_rmse),
  data_frame(method = "Movie + User + Genres + Days + Release year Effects Model", 
             RMSE = model_6_rmse),
  data_frame(method = "Regularized Movie + User + Genres + Days + Release year Effects Model",
             RMSE = min(rmses)))

rmse_results %>% knitr::kable()

