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
MAX_LEN = 1024
random.seed(SEED)

# format is: Tuple[link, pointer_for_posts, pointer_for_sentences]
links_and_divs = [
    ("https://www.tonightsbedtimestory.com/stories/", {"class" : "post"}, {"class" : "body"}),
]
for page_no in range(1, 25): # have to do it this way bcos this website has 24 pages
    link = "https://www.studentuk.com/category/bedtime-stories/page/" + str(page_no)
    links_and_divs.append((link, {"class" : "column margin50"}, {"class" : "entry-content"}))

def scrape_one_link(link_n_div):
    # print(f'Scraping {link_n_div[0]}')
    req = Request(link_n_div[0], headers={'User-Agent': 'Mozilla/5.0'}) # Only looking at home page

    page = urlopen(req).read()
    soup = BeautifulSoup(page, features="html.parser")
    posts = soup.findAll("div", link_n_div[1]) # pointer for posts

    # passage = ""
    counter = 0
    all_passages = []
    for post in tqdm(posts, desc=f'Scraping {link_n_div[0]}'):
        counter += 1
        curr_passage = ""
        url = post.a['href']
        if "javascript" in url: # Removing interactive and videos
            continue
        # print(url)
        currReq = Request(url,headers={'User-Agent': 'Mozilla/5.0'})
        pageLink = urlopen(currReq).read()
        soupLink = BeautifulSoup(pageLink, features="html.parser")
        sentences = soupLink.findAll("div", link_n_div[2]) # pointer for sentences
        for sentence in sentences:
            curr_passage += sentence.text
        if curr_passage:
            # remove some special characters
            curr_passage = curr_passage.replace('\n',' ')
            curr_passage = curr_passage.replace('\xa0','')

            curr_passage = curr_passage.replace('Mr.', 'Mr,')
            curr_passage = curr_passage.replace('Mrs.', 'Mrs,')
            curr_passage = curr_passage.replace('Ms.', 'Ms,')

            # curr_passage = "<BOS> " + curr_passage + " <EOS>"
            start, end = 0, MAX_LEN
            trimmed = ""
            curr_len = len(curr_passage)
            banned = set(["”", "’"]) # if fullstop is immediately followed by either of these, the sentence hasn't ended yet
            while curr_len > MAX_LEN:
                if curr_passage[end - 1] == '.' and curr_passage[end] not in banned:
                    trimmed += "<BOS> " + curr_passage[start:end] + " <EOS>\n"
                    curr_passage = curr_passage[end:]
                    curr_len = len(curr_passage)
                    end = MAX_LEN
                else:
                    end -= 1

            if curr_passage:
                trimmed += "<BOS> " + curr_passage + " <EOS>\n"
            trimmed = trimmed.replace("><", "> <")
            trimmed = trimmed.replace('Mr,', 'Mr.')
            trimmed = trimmed.replace('Mrs,', 'Mrs.')
            trimmed = trimmed.replace('Ms,', 'Ms.')
            all_passages.append(trimmed)
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

    text_file = open(f"data/train_{MAX_LEN}_n_{int(TRAIN_RATIO*100)}.txt", "w")
    text_file.write(' '.join(train))
    text_file.close()

    text_file = open(f"data/eval_{MAX_LEN}_n_{int(TRAIN_RATIO*100)}.txt", "w")
    text_file.write(' '.join(eval))
    text_file.close()

    print(f'Total time elapsed: {time.time() - start}')
    print('Finished saving train & eval text files in ./data/')

if __name__ == '__main__':
    main() 
    # old code: without splitting into 1024-char sentences
    # taks 50 seconds for 24+1 websites, total 288+79 = 367 stories, parallelized on 16 cores

    # new code: with splitting (backtracking algorithm)
    # takes 48 seconds for 24+1 websites, total 288+79 = 367 stories, parallelized on 24 cores. Nice!
    # got total of ~3.8k sequences (each ~1024 characters long and are in complete sentences)