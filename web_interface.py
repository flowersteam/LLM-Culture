from flask import Flask, request, render_template

from dummy_main_analysis import main_analysis

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        directory = request.form.get('dir')
        ticks_font_size = int(request.form.get('ticks_font_size'))
        labels_font_size = int(request.form.get('labels_font_size'))
        title_font_size = int(request.form.get('title_font_size'))

        analyzed_dir = f"Results/{directory}"
        font_sizes = {'ticks': ticks_font_size,
                      'labels': labels_font_size,
                      'title': title_font_size}

        
        print(f"\nLaunching analysis on the {analyzed_dir} results")
        main_analysis(analyzed_dir, font_sizes, plot=False)

        # The return statement will only be executed when the plots are generated, so can already 
        # redirect to new page with the figures
        return 'analysis running ... TODO : generate and plot the figures automatically'

    return render_template('analysis.html')

if __name__ == "__main__":
    app.run(debug=True)
