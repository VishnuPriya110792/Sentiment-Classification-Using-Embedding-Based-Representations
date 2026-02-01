with open('Tweets.csv', 'r', encoding='utf-8') as f:
    for i in range(5):
        line = f.readline()
        if not line:
            break
        print(repr(line))
print('---done---')
