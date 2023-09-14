from meridian_utils import compare

main_folder = r'C:\Users\kzammit\Documents\Detector\20230913'
test_annot_csv = main_folder + '\\' + r'inputs\annotations_test.csv'
detections = main_folder + '\detections.csv'

comparison = compare(test_annot_csv, detections)

comparison.to_excel(main_folder + r'\detected_annotations.xlsx')

