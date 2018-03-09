# Determines recommendations for users based on collaborative filtering

import math
from data import data

# euclidean similarity algorithm
def euclidean_similarity(user1, user2):
  # get items both users have ranked
  mutual_rankings = [item for item in data[user1] if item in data[user2]]
  # get their rankings for their mutual ranked items aas a pair
  rankings = [(data[user1][item], data[user2][item]) for item in mutual_rankings]
  # now we can find the distance between each pair of rankings
  distance = [(rank[0] - rank[1])**2 for rank in rankings]
  return 1 / (1 + sum(distance))

# pearson similarity algorithm
def pearson_similarity(user1, user2):
  # items both users have ranked
  mutual_rankings = [item for item in data[user1] if item in data[user2]]
  n = len(mutual_rankings)
  # find the sum of each of users ranking
  user1_sum = sum([data[user1][item] for item in mutual_rankings])
  user2_sum = sum([data[user2][item] for item in mutual_rankings])
  # find the sum of each ranking squared
  ss1 = sum([pow(data[user1][item], 2) for item in mutual_rankings])
  ss2 = sum([pow(data[user2][item], 2) for item in mutual_rankings])
  # sum of each of users ranking * other users ranking
  ps = sum([data[user1][item] * data[user2][item] for item in mutual_rankings])

  # find correlation coefficent
  numerator = n * ps - ( user1_sum * user2_sum )
  denominator = math.sqrt((n * ss1 - math.pow(user1_sum, 2)) * (n * ss2 - math.pow(user2_sum, 2)))
  return (numerator / denominator) if denominator != 0 else 0

def recommend(user, bound, similarity_fn):
  # find scores relative to every other user
  scores = [(similarity_fn(user, other), other) for other in data if other != user]

  scores.sort()
  scores.reverse()
  # only consider bound # of scores
  scores = scores[0:bound]

  # dictionary of recommendations - per other users
  recommendations = {}
  # build dictionary - estimates users item rank based off similar users
  for similarity, other in scores:
    ranked = data[other]
    for item in ranked:
      if item not in data[user]:
        weight = similarity * ranked[item]
        if item in recommendations:
          s, weights = recommendations[item]
          recommendations[item] = (s + similarity, weights + [weight])
        else:
          recommendations[item] = (similarity, [weight])

  for r in recommendations:
    similarity, item = recommendations[r]
    recommendations[r] = sum(item) / similarity

  return recommendations


# Lets get some recommendations and compare the two similarity algorithms
bound = 5
user = 'Cersei Lannister'

print(recommend(user, bound, euclidean_similarity))
print(recommend(user, bound, pearson_similarity))
