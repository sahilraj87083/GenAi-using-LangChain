from langchain.schema.runnable import RunnableLambda

def wordCounter(text):
    return len(text.split())

runnable_word_counter = RunnableLambda(wordCounter)

print(runnable_word_counter.invoke("you are a joke!"))

# it transforms any python function to a runnable so that it can be a part of any chain