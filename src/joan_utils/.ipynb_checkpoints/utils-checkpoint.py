from .clusters_utils import ListDict
from tqdm import tqdm 
import re
import emoji


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def split_verdicts_comments_amit(author_comments):
    author_verdicts = ListDict()
    author_otherComments = ListDict()
    
    nta_keywords = ['nta', 'you are not the asshole', 'you\'re not the asshole', 'u are not the asshole', 'u re not the asshole', 
                    'you re not the asshole', 'u\'re not the asshole', 'not the asshole', 'not the AH', 'not asshole', 'not AH']
    yta_keywords = ['yta', 'you are the asshole', 'you\'re the asshole', 'u are the asshole', 'u re the asshole', 
                    'you re the asshole', 'u\'re the asshole', 'you the AH', 'you the asshole', 'u the asshole', 'u the AH']

    for author, comments in tqdm(author_comments.items(), desc="Using keywords to extract verdicts"):
        for comment in comments:
            flag = False
            text = comment['body'].strip().lower()
            for key in (nta_keywords + yta_keywords):
                if key in text and comment['distinguished'] != 'moderator':
                    author_verdicts.append(author, comment['body'].strip())
                    flag = True
                    break
            
            if not flag and comment['distinguished'] != 'moderator':
                author_otherComments.append(author, comment['body'].strip())
                
    return author_otherComments, author_verdicts
                

EMOJI_DESCRIPTION_SCRUB = re.compile(r':(\S+?):')
HASHTAG_BEFORE = re.compile(r'#(\S+)')
BAD_HASHTAG_LOGIC = re.compile(r'(\S+)!!')
FIND_MENTIONS = re.compile(r'@(\S+)')
LEADING_NAMES = re.compile(r'^\s*((?:@\S+\s*)+)')
TAIL_NAMES = re.compile(r'\s*((?:@\S+\s*)+)$')

def process_tweet(s, save_text_formatting=True, keep_emoji=False, keep_usernames=False):
    if save_text_formatting:
        s = re.sub(r'https\S+', r'', str(s))
        s = re.sub(r'http\S+', r'', str(s))
    else:
        s = re.sub(r'http\S+', r'', str(s))
        s = re.sub(r'https\S+', r' ', str(s))
        s = re.sub(r'x{3,5}', r' ', str(s))
    
    s = re.sub(r'\\n', ' ', s)
    s = re.sub(r'\s', ' ', s)
    s = re.sub(r'<br>', ' ', s)
    s = re.sub(r'&amp;', '&', s)
    s = re.sub(r'&#039;', "'", s)
    s = re.sub(r'&gt;', '>', s)
    s = re.sub(r'&lt;', '<', s)
    s = re.sub(r'\'', "'", s)

    if save_text_formatting:
        s = emoji.demojize(s)
    elif keep_emoji:
        s = emoji.demojize(s)
        s = s.replace('face_with', '')
        s = s.replace('face_', '')
        s = s.replace('_face', '')
        s = re.sub(EMOJI_DESCRIPTION_SCRUB, r' \1 ', s)
        s = s.replace('(_', '(')
        s = s.replace('_', ' ')

    s = re.sub(r"\\x[0-9a-z]{2,3,4}", "", s)
    
    if save_text_formatting:
        s = re.sub(HASHTAG_BEFORE, r'\1!!', s)
    else:
        s = re.sub(HASHTAG_BEFORE, r'\1', s)
        s = re.sub(BAD_HASHTAG_LOGIC, r'\1', s)
    
    if save_text_formatting:
        #@TODO 
        pass
    else:
        # If removing formatting, either remove all mentions, or just the @ sign.
        if keep_usernames:
            s = ' '.join(s.split())

            s = re.sub(LEADING_NAMES, r' ', s)
            s = re.sub(TAIL_NAMES, r' ', s)

            s = re.sub(FIND_MENTIONS, r'\1', s)
        else:
            s = re.sub(FIND_MENTIONS, r' ', s)
    #s = re.sub(re.compile(r'@(\S+)'), r'@', s)
    user_regex = r".?@.+?( |$)|<@mention>"    
    s = re.sub(user_regex," @user ", s, flags=re.I)
    
    # Just in case -- remove any non-ASCII and unprintable characters, apart from whitespace  
    s = "".join(x for x in s if (x.isspace() or (31 < ord(x) < 127)))
    s = ' '.join(s.split())
    return s
                
                