#recipe recommender
import pandas as pd
import numpy as np
import heapq
import pickle
import os

filename = 'model/recipEeze_recommender_model_SMALL.sav'
item_based_recommender = pickle.load(open(filename, 'rb'))

recipes = pd.read_csv('datasets/rr-recipes.csv')
ratings = pd.read_csv('datasets/ratings-s.csv')

def get_r(user_id):
    # Select which system to use. Due to memory constraints, item based is the only viable option
    recommender_system = item_based_recommender
    # N will represent how many items to recommend
    N = 1000

    # The setting to a set and back to list is a failsafe.
    rated_items = list(set(ratings.loc[ratings['userid'] == user_id]['itemid'].tolist()))
    ratings_list = recipes['itemid'].values.tolist()
    reduced_ratings = ratings.loc[ratings['itemid'].isin(ratings_list)].copy()

    # Self explanitory name
    all_item_ids = list(set(reduced_ratings['itemid'].tolist()))

    # New_items just represents all the items not rated by the user
    new_items = [x for x in all_item_ids if x not in rated_items]

    # Estimate ratings for all unrated items
    predicted_ratings = {}
    for item_id in new_items:
        predicted_ratings[item_id] = recommender_system.predict(user_id, item_id).est
        pass

    # Get the item_ids for the top ratings
    recommended_ids = heapq.nlargest(N, predicted_ratings, key=predicted_ratings.get)
    recommended_ids = sorted(recommended_ids)

    # predicted_ratings
    recommended_df = recipes.loc[recipes['itemid'].isin(recommended_ids)].copy()
    #recommended_df.insert(1, 'pred_rating', np.zeros(len(recommended_ids)))
    recommended_df.insert(1, 'pred_rating', 0)

    # recommended_df = recipes.copy()
    for idx,item_id in enumerate(recommended_ids):
        recommended_df.iloc[idx, recommended_df.columns.get_loc('pred_rating')] = predicted_ratings[item_id]
        pass
    return recommended_df.head(N).sort_values('pred_rating', ascending=False)

def cap_str(item):
    string = item
    return string.capitalize()

def reg_frame(r_list,items):
    s_ = ''
    for i in items:
        j = i.strip()
        str_ = f'(?=.*{j})'
        s_ += str_
    s_
    r_list = r_list[r_list['ingredients'].str.contains(fr'^\b{s_}\b',regex=True)]
    return r_list

def set_up_ml(user_id,ingredient_list):
    recipe_list = get_r(user_id)
    items = ingredient_list.split(',')
    recipe_list = reg_frame(recipe_list,items)
    return recipe_list

def get_final_recommendation(list_1,list_2,list_3): # combine all recommendations
    recipe_recommendation = pd.DataFrame()
    recipe_recommendation = pd.concat([list_1,list_2,list_3]) # concat lists
    recipe_recommendation = recipe_recommendation.drop_duplicates() # drop recommended duplicates of films
    recipe_recommendation = recipe_recommendation.sort_values('pred_rating',ascending=False) # sort by predicted rating
    recipe_recommendation.pop('pred_rating') # drop the rating column
    recipe_recommendation = recipe_recommendation.reset_index()
    recipe_recommendation.pop('index') # reset and pop the old index
    recipe_recommendation.pop('itemid')
    #recipe_recommendation['total time'] = recipe_recommendation['prep_time']+recipe_recommendation['cook_time']
    return recipe_recommendation
