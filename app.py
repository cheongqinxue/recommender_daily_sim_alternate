import streamlit as st
st.set_page_config(layout="wide")
import faiss
from faiss import METRIC_INNER_PRODUCT
import pandas as pd
import joblib
import numpy as np
import torch
from bipartite_models import TransRBipartiteModel
import requests
import plotly.graph_objects as go
import s3fs
import logging
import math

logger = logging.getLogger(__name__)

class Args:
    datapath = st.secrets['DATAPATH']
    modelpath = st.secrets['MODELPATH']
    
FS = s3fs.S3FileSystem(anon=False)

@st.cache
def load(path):
    if FS is None:
        df = joblib.load(path+'/df.joblib')
        emb = np.load(path+'/embeds.npy')
        domain_media = joblib.load(path+'/domain_media_df.joblib')
    else:
        with FS.open(path+'/df.joblib') as f:
            df = joblib.load(f)
        with FS.open(path+'/embeds.npy') as f:
            emb = np.load(f)
        with FS.open(path+'/domain_media_df.joblib') as f:
            domain_media = joblib.load(f)

    string_factory = 'IVF256,Flat'
    print('Building index...', end='')
    index = faiss.index_factory(384, string_factory, METRIC_INNER_PRODUCT)    
    index.train(emb)
    index.add(emb)
    index.nprobe = 12

    return df, domain_media, emb, index


def search(domain, rep_vectors, faiss_index, df, head2ix, embeddings, model, display_top_n=20, 
    search_n_per_signpost=5000, language='any', debug=False, favor='na', sensitivity=0.4):

    reps = torch.vstack(rep_vectors['rep_vectors'][domain])

    if len(favor) > 0:
        favor = [int(sn) for sn in favor]

        k = display_top_n*30 if language != 'any' else math.ceil(display_top_n/len(favor))

        scores, indices = faiss_index.search(embeddings[favor,:], display_top_n*20)

        if language != 'any':
            tmp = []
            for indices_ in indices:
                indices_ = indices_.tolist()
                languages = df.loc[indices_,:]['language'].tolist()
                indices_ = [ix for ix, l in zip(indices_, languages) if str(l) in language]
                indices_ = indices_[:math.ceil(display_top_n/len(favor))]
                tmp.append(indices_)
            indices = np.asarray(tmp)
        else:
            tmp = []
            for indices_ in indices:
                indices_ = indices_.tolist()
                languages = df.loc[indices_,:]['language'].tolist()
                indices_ = [ix for ix, l in zip(indices_, languages)]
                indices_ = indices_[:math.ceil(display_top_n/len(favor))]
                tmp.append(indices_)
            indices = np.asarray(tmp)

        indices = [ix for ix in indices.reshape(-1).tolist() if ix not in favor]
        

        # index_set = []
        # for ix in indices:
        #     if ix not in index_set:
        #         index_set.append(ix)
        # indices = index_set[:display_top_n]
    else:
        _, indices = faiss_index.search(reps.numpy(), search_n_per_signpost)  
        indices = list(set(indices.reshape(-1).tolist()))
        if language != 'any':
            languages = df.iloc[indices,:]['language'].tolist()
            indices = [ix for ix, l in zip(indices, languages) if str(l) in language]

    with torch.no_grad():
        h = head2ix[domain]
        te = torch.tensor(embeddings[indices], device='cpu')
        scores = model.scoring_function(
                h_idx=torch.tensor([h], device = 'cpu'),
                r_idx=torch.tensor([0], device = 'cpu'),
                t_idx=None,
                new_tails=te)
        scores = torch.tanh(scores+2.5)
        topn = torch.argsort(scores, descending=True)[:max(300, int(search_n_per_signpost/4))].tolist()

    indices_ = np.asarray(indices)[topn].tolist()
    scores_ = scores[topn].numpy().tolist()
    resultdf = df.iloc[indices_,:].drop(columns=['media_item_id'])
    resultdf['score'] = scores_
    resultdf = resultdf.drop_duplicates(subset='title')
    if language != 'any':
        resultdf = resultdf[resultdf.language==language]
    resultdf = resultdf.drop(columns=['language'])
    try:
        resultdf = resultdf.head(display_top_n)
        resultdf['title'] = resultdf.title.apply(lambda x: x[:120])
        resultdf['content'] = resultdf.content.apply(lambda x: x[:200]+'...')
        return resultdf
    except Exception as e:
        print('topn ', topn[:10])
        print('indices ', indices[:10])
        if debug:
            raise(e)
        else:
            print(e)
            return topn, indices
    return

def render(container, container2, domain_media_df, **kwargs):
    resultdf = search(**kwargs)

    if resultdf is None:
        raise "search failed"
    else:
        fig = go.Figure(
            data=[
                go.Table(
                    columnwidth=[50,270,450,100,50],
                    header=dict(values=['Type','Title','Content','Date','Score'],
                        fill_color='lightsteelblue',
                        font_color='black',
                        font_size=15,
                        align='left'),
                    cells=dict(values=[resultdf.type, resultdf.title, resultdf.content, resultdf.published_date.dt.strftime('%Y-%m-%d %H:%M'), resultdf.score.round(3)],
                        fill_color='#EEEEEE',
                        font_size=13,
                        align='left')
                )
            ])
        fig.update_layout(
            margin=dict(l=20, r=20, t=5, b=5),height=250)
        container.plotly_chart(fig, use_container_width=True)

        ddf = domain_media_df[domain_media_df.name==kwargs['domain']]
        ddf = ddf.merge(kwargs['df'], how='left', on='media_item_id')
        if kwargs['language']!='any':
            ddf = ddf[ddf.language==kwargs['language']]
        ddf = ddf.head(50)
        fig = go.Figure(
            data=[
                go.Table(
                    columnwidth=[50,50,220,450,150],
                    header=dict(values=['Sn','Type','Title','Content','Date'],
                        fill_color='lightsteelblue',
                        font_color='black',
                        font_size=15,
                        align='left'),
                    cells=dict(values=[ddf['index'], ddf.type, ddf.title, ddf.content.apply(lambda x: x[:500]+'...'), ddf.published_date.dt.strftime('%Y-%m-%d %H:%M')],
                        fill_color='#EEEEEE',
                        font_size=13,
                        align='left')
                )
            ])
        fig.update_layout(
            margin=dict(l=20, r=20, t=5, b=5),height=550)
        container2.plotly_chart(fig, use_container_width=True)


def main(args):
    df, domain_media_df, embeddings, index = load(args.datapath)
    languages = ['any','en','es','pt'] + sorted(list(df.dropna(subset=['language']).language.unique()))

    model, head2ix = TransRBipartiteModel.load_pretrained(args.modelpath, fh=FS)

    if FS is None:
        with open(args.modelpath+'/rep_vectors.pt', 'rb') as f:
            rep_vectors = torch.load(f)
    else:
        with FS.open(args.modelpath+'/rep_vectors.pt') as f:
            rep_vectors = torch.load(f)

    du = st.sidebar.selectbox(label = 'Select your domain unit', options=sorted(list(head2ix.keys())), 
        index=0, key=None, help=None)

    lang = st.sidebar.selectbox(label = 'Select your preferred language', options=languages)

    sn = st.sidebar.text_input(label = 'Enter the serial numbers of news from the daily listing to simulate in-session reading activity', 
                               help = 'To simulate reading of one or more news articles, enter the serial number of the article listed under the "Sn" column separated by a comma. E.g.225210,175694')
    
    # sensitivity = st.sidebar.select_slider('Choose how sensitive the recommender will be to in-session reading activity (higher = more sensitive)', 
    #                                         options=[i/20 for i in range(21)], value=0.5)
    
    st.sidebar.markdown('#### In-session Recommendations Simulator')
    explanation = ('Use the text-inputs above to simulate your own recent reading history. Enter the serial numbers of articles from the daily panel '
                   'to find other articles like them. For instance, enter "2835" to find more platinium-mining related news. \n A longer reading history list will prompt the recommender to give more weight to the recent '
                   'reading history when searching for articles. All articles come from a 3-day (22 Nov to 24 Nov) window.'
                  )
    st.sidebar.markdown(explanation)
    
    c1 = st.container()
    c1.subheader('Recommended Articles')
    c2 = st.container()
    c2.subheader('Daily Articles [As of 24 Nov 21]')
    sn = [s.strip() for s in sn.split(',')]
    if all([s.isnumeric() for s in sn]):
        sn = [int(s) for s in sn]
        logger.info(df.loc[sn].title)
    else:
        sn = []
    
    render(container = c1, container2=c2, domain_media_df=domain_media_df, **{'domain':du, 'rep_vectors':rep_vectors, 'faiss_index':index, 'df':df, 
        'head2ix':head2ix, 'embeddings':embeddings, 'model':model, 'language':lang, 'favor':sn})


if __name__ == '__main__':
    args = Args()
    
    pwd_ = st.sidebar.text_input(label='Enter Password')

    if pwd_ == st.secrets['PASSWORD']:
        main(args)
    else:
        st.error('Wrong password')
        st.stop()
