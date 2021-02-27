import random
import multiprocessing
import os
import itertools
import time

from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from tqdm import tqdm

SEED = 1337
TRAIN_RATIO = 0.8 # EVAL_RATIO = 1 - TRAIN_RATIO
random.seed(SEED)

# format is: Tuple[link, pointer_for_posts, pointer_for_sentences]
links_and_divs = [
    ("https://www.tonightsbedtimestory.com/stories/", {"class" : "post"}, {"class" : "body"}),
]
for page_no in range(1, 25): # have to do it this way bcos this website has 24 pages
    link = "https://www.studentuk.com/category/bedtime-stories/page/" + str(page_no)
    links_and_divs.append((link, {"class" : "column margin50"}, {"class" : "entry-content"}))

def scrape_one_link(link_n_div):
    print(f'Scraping {link_n_div[0]}')
    req = Request(link_n_div[0], headers={'User-Agent': 'Mozilla/5.0'}) # Only looking at home page

    page = urlopen(req).read()
    soup = BeautifulSoup(page, features="html.parser")
    posts = soup.findAll("div", link_n_div[1]) # pointer for posts

    # passage = ""
    counter = 0
    all_passages = []
    for post in posts: #desc='Scraping each story'):
        counter += 1
        # curr_passage = "<BOS> "
        curr_passage = ""
        url = post.a['href']
        if "javascript" in url: # Removing interactive and videos
            continue
        print(url)
        currReq = Request(url,headers={'User-Agent': 'Mozilla/5.0'})
        pageLink = urlopen(currReq).read()
        soupLink = BeautifulSoup(pageLink, features="html.parser")
        sentences = soupLink.findAll("div", link_n_div[2]) # pointer for sentences
        for sentence in sentences:
            curr_passage += sentence.text
        if curr_passage:
            curr_passage = "<BOS> " + curr_passage + " <EOS>"

            # remove some special characters
            curr_passage = curr_passage.replace('\n',' ')
            curr_passage = curr_passage.replace('\xa0','')
            curr_passage = curr_passage.replace('><','> <')
            
            all_passages.append(curr_passage)
    return all_passages

def main():
    start = time.time()

    passages = []
    num_cores = len(os.sched_getaffinity(0))
    print(f'Parallelizing over {num_cores} cores')
    pool = multiprocessing.Pool(num_cores)
    for result in tqdm(pool.imap(scrape_one_link, links_and_divs),
            total=len(links_and_divs), desc='Scraping each link'):
        passages.append(result)

    passages = list(itertools.chain.from_iterable(passages))
    print(f'Total stories scraped: {len(passages)}')

    # split into train & eval
    random.shuffle(passages)
    train = passages[:int(TRAIN_RATIO * len(passages))]
    eval = passages[int(TRAIN_RATIO * len(passages)):]

    text_file = open("data/train.txt", "w")
    text_file.write(' '.join(train))
    text_file.close()

    text_file = open("data/eval.txt", "w")
    text_file.write(' '.join(eval))
    text_file.close()

    print(f'Total time elapsed: {time.time() - start}')

print('Finished saving as train.txt & eval.txt')

if __name__ == '__main__':
    main() # taks 50 seconds for 24+1 websites, total 288+79 = 367 stories