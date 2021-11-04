# Question Answering With Google's T5 Transformer
If you want to learn what is T5, you can read [this blog.](https://medium.com/syncedreview/google-t5-explores-the-limits-of-transfer-learning-a87afbf2615b) <br/>
In this project, I have applied two diffferent question answering task: **Open book question** and **Closed book question** <br/>
### Open Book Question 
Open book question means that model is given a textual information which is called **context** <br/>
Then, model answers questions according to the information on the given context. <br/>
### Closed Book Question
Closed book question means the opposite of open book question, so model is not given a textual information. <br/>
So, how model answer questions? <br/>
In T5's terms this means that T5 can only use it's stored weights to answer a question and is given no aditional context. <br/>
T5 was pre-trained on the C4 dataset which contains petabytes of web crawling data collected over the last 8 years, including Wikipedia in every language. <br/>
This gives T5 the broad knowledge of the internet stored in it's weights to answer various closed book questions <br/>



You can see the notebook in ***question_answering.ipynb*** file. 


