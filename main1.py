import praw
import pandas as pd
import time
from prawcore.exceptions import RequestException

# Set up the Reddit API client
reddit = praw.Reddit(
    client_id='DZN8fwVqoCCpoYRfraQB2w', 
    client_secret='43fzFXo_MawKif395JyM7HtBVxK-7w', 
    user_agent='Jayanti Dwivedi',
    timeout=30  # Extended timeout to avoid frequent timeouts
)

# Define the subreddits to scrape
subreddits = [
    
     
     'imsorryjon',
    'desirepath',
     'Backrooms',                # For eerie, unfinished-looking real-world places.
    'LiminalSpace',             # Images evoking a sense of in-betweenness.
    'BoneHurtingJuice',         # Anti-humor subreddit with literal meme edits.
    'DisneyVacation',           # Wacky captions on strange wikiHow illustrations.
    'BootTooBig',               # "Roses are red" poems using images as punchlines.
    'Slavs_Squatting',          # Humor around Eastern European stereotypes.
    'BreadStapledToTrees',      # Photos of bread stapled to trees.
    'BirdsArentReal',           # Satirical conspiracy that birds are government drones.
    'BirdsWithArms',            # Photoshopped images of birds with human arms.
    'SubSimulatorGPT2',
]
    




# Define the function to scrape posts and comments
def scrape_reddit(subreddits, limit_per_subreddit=100):
    posts_data = []
    comments_data = []

    for subreddit in subreddits:
        print(f"Scraping subreddit: {subreddit}")
        try:
            submission_list = reddit.subreddit(subreddit).new(limit=limit_per_subreddit)
        except RequestException as e:
            print(f"Failed to fetch subreddit {subreddit}: {e}. Retrying...")
            time.sleep(5)
            continue

        for submission in submission_list:
            # Get the post data
            post = {
                'post_id': submission.id,
                'title': submission.title,
                'body': submission.selftext,
                'subreddit': submission.subreddit.display_name,
                'flair': submission.link_flair_text,
                'upvotes': submission.score,
                'downvotes': submission.downs,
                'total_score': submission.score,
                'timestamp': submission.created_utc,
                'awards': submission.total_awards_received,
                'num_comments': submission.num_comments,
                'media_type': submission.url if submission.url else 'text',
            }
            posts_data.append(post)

            # Fetch comments safely
            try:
                submission.comments.replace_more(limit=0)  # Avoid fetching "MoreComments" objects
                for comment in submission.comments.list()[:10]:  # Limit to 10 comments per post
                    comments_data.append({
                        'post_id': submission.id,
                        'comment_id': comment.id,
                        'author': comment.author.name if comment.author else 'deleted',
                        'body': comment.body,
                        'upvotes': comment.score,
                        'timestamp': comment.created_utc,
                    })
            except RequestException as e:
                print(f"Error fetching comments for post {submission.id}: {e}")
                time.sleep(5)  # Retry delay

        # Add delay between requests to avoid rate-limiting
        time.sleep(2)
    
    # Convert data to pandas DataFrame
    posts_df = pd.DataFrame(posts_data)
    comments_df = pd.DataFrame(comments_data)

    return posts_df, comments_df

# Main execution
if __name__ == '__main__':
    try:
        posts_df, comments_df = scrape_reddit(subreddits)
        # Save the data to CSV
        posts_df.to_csv('reddit_posts10.csv', index=False)
        comments_df.to_csv('reddit_comments10.csv', index=False)
        print("Data saved to CSV files.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
