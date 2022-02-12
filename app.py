import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# define data frame
if 'df' not in st.session_state:
    df = pd.DataFrame()
else:
    df = st.session_state['df']



st.header('Config setting')

# seed
with st.expander('seed'):
    seed = st.slider('random seed', 0, 9999, step=1)

def init_seed(seed):
    np.random.seed(seed)
init_seed(seed)


# data size
data_n = 100

with st.expander('data size'):
    col1, col2 = st.columns(2)
    with col1:
        data_n  = st.slider('', 1, 1_000_000, value=data_n, step=1)
    with col2:
        data_n  = st.text_input('', value=data_n)
        data_n = int(data_n)

if 'data_n' in st.session_state:
    del st.session_state['df']

# output file name
with st.expander('output file name'):
    output_fname = st.text_input('', value='train.csv')




st.header('Genertor')

# digit
with st.expander('Gaussian noise'):
    gn_col_name = st.text_input('column name', value='col1')
    gn_mean = st.slider('gaussian noise mean', -99., 99., value=0.)
    gn_std  = st.slider('gaussian noise std', -99., 99., value=1.)
    gn = np.random.normal(gn_mean, gn_std, size=data_n)
    df[gn_col_name] = gn

    # show sample
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.hist(gn, bins=20)
    ax.grid()
    st.pyplot(fig)

# category
with st.expander('Category'):
    cat_col_name = st.text_input('category column name', value='category1')
    cat_input = st.text_area('input category(, weight)   if weight is none, weight=1', value="""Newt,1\nfrog,2\nturtle,4""")

    def category_parser(text):
        cats = {}
        sums = 0.

        for line in text.split('\n'):
            line = line.strip()
            if ',' in line:
                c, w = line.split(',')
                c = c.strip()
                w = w.strip()
                if w == '':
                    w = 1
            else:
                c = line.strip()
                w = 1
            w = float(w)
            sums += w
            cats[c] = w

        for c, w in cats.items():
            cats[c] /= sums

        return cats

    def generate_cat_values(cats, data_n):
        vals = []
        res = {}

        for c, w in cats.items():
            cn = int(np.round(data_n * w))
            vals += [c] * cn
            res[c] = cn

        np.random.shuffle(vals)

        if len(vals) > data_n:
            vals = vals[:data_n]

        if len(vals) < data_n:
            vals += vals[:data_n - len(vals)]

        return vals, res

    cats = category_parser(cat_input)
    cat_vals, result = generate_cat_values(cats, data_n)
    df[cat_col_name] = cat_vals

    # show sample
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.grid()
    ax.hist(df[cat_col_name])
    # ax.barh(list(result.keys()), list(result.values()))
    st.pyplot(fig)


# delete column
with st.expander('[Warning] Delete column'):
    delete_cols = st.multiselect('', df.columns)
    for delete_col in delete_cols:
        del df[delete_col]

# save state
st.session_state['df'] = df
st.session_state['seed'] = seed
st.session_state['data_n'] = data_n

# config
st.sidebar.title('Config')
st.sidebar.text(f'data size   = {data_n}')
st.sidebar.text(f'seed        = {seed}')
st.sidebar.text(f'output file = {output_fname}')

# output
st.header('Output csv')
df

@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')

st.download_button('Press to Download', convert_df(df), output_fname, 'text/csv', key='download-csv')


# show column
with st.expander('Plot distribution'):
    select_col = st.selectbox('', df.columns)

    if df[select_col].dtype is not object:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.hist(df[select_col], bins=20)
        ax.grid()
        st.pyplot(fig)

    else:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.hist(df[select_col], bins=20)
        ax.grid()
        st.pyplot(fig)