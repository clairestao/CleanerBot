import discord
import os

import numpy as np
import pandas as pd

import tensorflow as tf

from dotenv import load_dotenv
load_dotenv()


LOAD_DATA = True

if (LOAD_DATA):
    model = tf.keras.models.load_model('model')

else:

    # training data from Wikipedia
    df = pd.read_csv("train.csv", usecols=['comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
    df["label"] = df.toxic | df.severe_toxic | df.obscene | df.threat | df.insult | df.identity_hate
    df = df[["comment_text", "label"]]

    train, val, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

    def df_to_dataset(dataframe, shuffle = True, batch_size = 1024):
        df = dataframe.copy()
        ds = tf.data.Dataset.from_tensor_slices((df["comment_text"], df.pop('label')))
        if shuffle:
            ds = ds.shuffle(buffer_size = len(dataframe))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_data = df_to_dataset(train)
    valid_data = df_to_dataset(val)
    test_data = df_to_dataset(test)

    encoder = tf.keras.layers.TextVectorization(max_tokens = 2000)
    encoder.adapt(train_data.map(lambda text, label: text))

    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim = encoder.vocabulary_size(),
            output_dim = 32,
            mask_zero = True
        ),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(32, activation = 'relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
                loss = tf.keras.losses.BinaryCrossentropy(),
                metrics = ['accuracy'])

    model.evaluate(train_data)
    model.evaluate(valid_data)
    history = model.fit(train_data, epochs = 5, validation_data = valid_data)
    model.evaluate(test_data)

    model.save('model')


client = discord.Client()

@client.event
async def on_ready():
    print("Logged in as {0.user}".format(client))

@client.event
async def on_message(message):

    # don't do anything if the message is by the bot
    if message.author == client.user:
        return
    
    # predict how likely it is to be offensive
    prediction = model.predict([message.content])[0][0]
    print(prediction)
    if prediction > 0.85:
        await message.author.send("Your message \'" + message.content + "\' has been marked as potentially offensive and has been removed.")
        await message.delete()

client.run(os.getenv('TOKEN'))