import csv

categories = ['Sports', 'Tech', 'Entertainment', 'Business', 'Travel']
texts = ['Latest scores and highlights on Golf!',
         'New gadgets and innovations in smart phone industry!',
         'Movie reviews and celebrity news from last week!',
         'Market trends and financial insights after rate hike!',
         'Exotic destinations and travel tips in Europe!']

with open('./data/sample_news_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Category', 'Text'])  # Write header row
    writer.writerows(zip(categories, texts))  # Write data rows efficiently

print('CSV file generated successfully!')
