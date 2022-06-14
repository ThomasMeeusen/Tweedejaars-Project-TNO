# Filter articles with keywords
Made by Taiki Papandreou

## What is this branch about?
This project is about classifying news articles. It aims to detect news articles that are related to workplace accidents. Our dataset contains 300 thousand news articles from CNN and Daily Mail. One of the first problems we faced was the small proportion of workplace accidents related articles in our dataset. We inspected the first 200 articles of the dataset and there were only few articles that were related to workplace accidents. <br>
However, accidents related articles have distinct vocabulary compared to economic or politics related articles. We can roughly filter the dataset to exclude articles that are remotely unrelated to accidents (e.g. politics, sports, celebrity, movie, art).

## The goal of this branch
- Exclude remarkably unrelated articles in order to obtain smaller dataset with higher proportion of accidents related articles.
- Detect articles that are related to workplace accidents.
- Detect a pattern of words that are used in workplace accidents related articles.

