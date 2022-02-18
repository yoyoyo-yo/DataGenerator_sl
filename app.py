import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

#---
# prepare data
#---
if 'gen_data' in st.session_state:
    gen_data = st.session_state['gen_data']
else:
    gen_data = {}


st.title('Data Generator')
st.markdown('You can generate random data. No data No Life')

st.header('Config setting')

#---
# seed
#---
with st.expander('seed'):
    seed = st.slider('random seed', 0, 9999, step=1)

def init_seed(seed):
    np.random.seed(seed)

init_seed(seed)

#---
# data size
#---
data_n = 100

with st.expander('data size'):
    col1, col2 = st.columns(2)
    with col1:
        data_n  = st.slider('', 1, 100_000, value=data_n, step=1)
    with col2:
        data_n  = st.text_input('', value=data_n)
        data_n = int(data_n)

if 'data_n' in st.session_state and data_n != st.session_state['data_n'] and 'gen_data' in st.session_state:
    del st.session_state['gen_data']
    gen_data = {}

# output file name
with st.expander('output file name'):
    output_fname = st.text_input('', value='train.csv')


#---
# Generator
#---

st.header('Genertor')

#---
# numeric
#---
with st.expander('Distribution'):

    dist_select = st.selectbox('Select distribution', ('Gaussian', 'Uniform', 'Uniform(int)'))

    st.caption('-'*20)
    
    num_col_name = st.text_input('Column name', value='col1')

    if dist_select == 'Gaussian':
        gn_mean = st.slider('Gaussian noise mean', -99., 99., value=0.)
        gn_std  = st.slider('Gaussian noise std',  0., 99., value=1.)
        num_vals = np.random.normal(gn_mean, gn_std, size=data_n)

    if dist_select == 'Uniform':
        values = st.slider('Select a range of values', -1000.0, 1000.0, (0.0, 100.0))
        num_vals = np.random.rand(data_n) * (float(values[1]) - float(values[0])) + float(values[0])

    if dist_select == 'Uniform(int)':
        values = st.slider('Select a range of values', -1000.0, 1000.0, (0.0, 100.0))
        num_vals = np.random.randint(float(values[0]), float(values[1]), data_n)
    

    if st.button('Shuffle', key='btn_shuffle_num'):
        random.shuffle(num_vals)

    # if st.button('OK! Add!', key='btn_add_gn'):
    gen_data[num_col_name] = num_vals

    # show sample
    st.markdown('***Sample***')
    st.text(num_vals[:min(5, data_n)])
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.hist(num_vals, bins=20)
    ax.grid()
    ax.set_title('distribution')
    st.pyplot(fig)

#---
# category
#---
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


with st.expander('Category'):
    cat_col_name = st.text_input('Category column name', value='category1')
    cat_input = st.text_area('input category(, weight)   if weight is none, weight=1', value="""Newt,1\nfrog,2\nturtle,4""")

    cats = category_parser(cat_input)
    cat_vals, result = generate_cat_values(cats, data_n)

    if st.button('Shuffle', key='btn_forceshuffle_cat'):
        random.shuffle(cat_vals)

    # if st.button('OK! Add!', key='btn_add_cat'):
    gen_data[cat_col_name] = cat_vals
    
    # show sample
    st.markdown('***Sample***')
    st.text(cat_vals[:min(5, data_n)])
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.grid()
    ax.hist(np.sort(cat_vals))
    ax.set_title('distribution')
    st.pyplot(fig)


#---
# delete column
#---
with st.expander('[Warning] Delete column'):
    delete_cols = st.multiselect('', gen_data.keys())
    _gen_data = gen_data.copy()

    for delete_col in delete_cols:
        del _gen_data[delete_col]

    _df = pd.DataFrame(_gen_data)
    st.caption('Result')
    st.dataframe(_df)

    if st.button('OK! Delete!', key='btn_delete'):
        for delete_col in delete_cols:
            del gen_data[delete_col]


#---
# output
#---
st.header('Output csv')

df = pd.DataFrame(gen_data)
st.dataframe(df)

@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')

st.download_button('Download!', convert_df(df), output_fname, 'text/csv', key='download-csv')

#---
# show column
#---
with st.expander('check data distribution'):

    if len(df.columns):
        select_col = st.selectbox('', df.columns)
        select_col

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

    else:
        st.caption('No data')


#---
# save state
#---
st.session_state['gen_data'] = gen_data
st.session_state['seed'] = seed
st.session_state['data_n'] = data_n

#---
# sidebar
#---
st.sidebar.title('***Usage***')
st.sidebar.caption('   1. Select data type')
st.sidebar.caption('   2. Check data')
# st.sidebar.text('  3. If you like, push "Add" button!')
# st.sidebar.header("***Don't forget enter after each input***")

st.sidebar.title('***Config***')
st.sidebar.caption(f'data size   = {data_n}')
st.sidebar.caption(f'seed        = {seed}')
st.sidebar.caption(f'output file = {output_fname}')

st.sidebar.title('***Output info***')
st.sidebar.caption(f'df size   = {df.shape}')
st.sidebar.subheader(f'***Column***')

for c in df.columns:
    st.sidebar.caption(f'{c}')