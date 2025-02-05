The Dyslexia Detection Game is an interactive web-based application designed to help detect signs of dyslexia. The game presents 10 audio-based questions 
where players hear a letter and must select the correct one from four visually similar options (e.g., hearing 'b' and choosing from 'b, d, p, q'). 
The collected responses are processed by a Flask backend, which utilizes a trained Machine Learning model (dyslexia_model.pkl) to
analyze user performance and predict possible dyslexia indicators.
Features
Audio-based letter recognition game
Frontend built with HTML, CSS, and JavaScript
Backend developed using Flask
Machine Learning model (dyslexia_model.pkl) for dyslexia detection
CSV data files (Dyt-desktop.csv and Dyt-tablet.csv) for processing and model evaluation
