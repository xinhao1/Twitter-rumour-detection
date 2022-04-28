# https://github.com/twitterdev/Twitter-API-v2-sample-code/blob/main/Tweet-Lookup/get_tweets_with_bearer_token.py
import requests
import json
import time
from tqdm import tqdm

# To set your bearer token:
bearer_token = "AAAAAAAAAAAAAAAAAAAAALJCbgEAAAAARxbc2VfIuOoyGTOkcrfRsBV%2Bj74%3DjgzWatKw8Ta69sHZlaYA2ZKqOmwfIUEq1r3oEKabzlcxBdFCMf"


def create_url(ids):
    tweet_fields = "tweet.fields=attachments,author_id,context_annotations,conversation_id,created_at,entities,geo,id,in_reply_to_user_id,lang,public_metrics,possibly_sensitive,referenced_tweets,reply_settings,source,text,withheld"
    # Tweet fields are adjustable.
    # Options include:
    # attachments, author_id, context_annotations,
    # conversation_id, created_at, entities, geo, id,
    # in_reply_to_user_id, lang, non_public_metrics, organic_metrics,
    # possibly_sensitive, promoted_metrics, public_metrics, referenced_tweets,
    # source, text, and withheld
    ids = "ids=" + ids
    # print(ids)
    # You can adjust ids to include a single Tweets.
    # Or you can add to up to 100 comma-separated IDs
    url = "https://api.twitter.com/2/tweets?{}&{}".format(ids, tweet_fields)
    return url


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2TweetLookupPython"
    return r


def connect_to_endpoint(url):
    response = requests.request("GET", url, auth=bearer_oauth)
    # print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()


def crawl_and_save(f_in, f_out):
    train_id_list = []
    for l in f_in.readlines():
        train_id_list.extend(l.strip().split(","))
    start_id = 0
    end_id = start_id + 100
    train_id_len = len(train_id_list)
    # max 100 tweet
    split_crawl = []
    while start_id < train_id_len:
        split_crawl.append(",".join(train_id_list[start_id:end_id]))
        start_id = end_id
        end_id = start_id + 100

    crawl_count = 0
    for ids in tqdm(split_crawl):
        url = create_url(ids)
        # url = create_url(",".join(["0", "1", "2", train_id_list[0]]))
        json_response = connect_to_endpoint(url)
        for x in json_response["data"]:
            json.dump(x, open(f_out + str(x["id"]) + ".json", "w"))
        crawl_count += 1
        if crawl_count % 290 == 0:
            time.sleep(1000)

def main():
    # print("crawl the train tweets")
    # crawl_and_save(open("data/train.data.txt", "r"), "data/train_tweet/")
    # print("crawl the dev tweets")
    # crawl_and_save(open("data/dev.data.txt", "r"), "data/dev_tweet/")
    print("crawl the analysis tweets")
    crawl_and_save(open("data/covid.data.txt", "r"), "data/analysis_tweet/")
    print("Finished!")

if __name__ == "__main__":
    main()
