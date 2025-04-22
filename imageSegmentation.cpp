#include <gtk/gtk.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>        
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <functional>   
#include <cmath>

using namespace cv;
using namespace std;

// Global variables and UI elements
GtkWidget *window;
GtkWidget *original_image_view;
GtkWidget *processed_image_view;
GtkWidget *algorithm_combo;
GtkWidget *apply_button;
GtkWidget *status_label;
GtkWidget *info_label;
GtkWidget *threshold_label;
GtkWidget *threshold_slider_box;
GtkWidget *threshold_slider;
GtkWidget *kmeans_slider_box;
GtkWidget *kmeans_slider;
char *filename = NULL;
Mat input_image;

// Algorithm parameters and thresholds
const int REGION_GROWING_THRESHOLD = 30;
const int ACTIVE_CONTOURS_ITERATIONS = 100;
const float ACTIVE_CONTOURS_ALPHA = 0.1;
const float ACTIVE_CONTOURS_BETA = 0.2;
const float ACTIVE_CONTOURS_GAMMA = 0.4;
const int KMEANS_MAX_ITER = 10;
const double KMEANS_EPSILON = 1.0;
const int WATERSHED_MORPH_SIZE = 3;
const int GRAPH_CUT_ITERATIONS = 5;
int BACKTRACKING_THRESHOLD = 128;
int KMEANS_CLUSTERS = 2;

// Forward declarations of segmentation algorithms
Mat activeContoursSegmentation(const Mat& image);
Mat kMeansSegmentation(const Mat& image, int clusters);
Mat otsuSegmentation(const Mat& image, double& otsuThreshold);
Mat backtrackingSegmentation(const Mat& image);
Mat backtrackingSegmentation8Dir(const Mat& image);
Mat backtrackingSegmentationImproved(const Mat& image);
Mat backtrackingEdgeEnhancementSegmentation(const Mat& image);
Mat watershedSegmentation(const Mat& image);
Mat graphCutSegmentation(const Mat& image);
Mat regionGrowingSegmentation(const Mat& image, Point seed, int threshold);

// Forward declarations
static void update_backtracking_segmentation();
static void update_backtracking_improved_segmentation();
static void update_backtracking_edge_enhanced_segmentation();
static void update_kmeans_segmentation();
static void on_algorithm_changed(GtkComboBox *widget, gpointer data);

// Callback for threshold slider change
static void on_threshold_changed(GtkRange *range, gpointer data) {
    BACKTRACKING_THRESHOLD = (int)gtk_range_get_value(range);
    
    // Only update if backtracking is selected
    gchar *selected_algorithm = gtk_combo_box_text_get_active_text(GTK_COMBO_BOX_TEXT(algorithm_combo));
    if (selected_algorithm != NULL) {
        if (strcmp(selected_algorithm, "Backtracking") == 0) {
            update_backtracking_segmentation();
        } else if (strcmp(selected_algorithm, "Backtracking Improved") == 0) {
            update_backtracking_improved_segmentation();
        } else if (strcmp(selected_algorithm, "Backtracking Edge Enhanced") == 0) {
            update_backtracking_edge_enhanced_segmentation();
        }
    }
    g_free(selected_algorithm);
}

// Callback for kmeans slider change
static void on_kmeans_changed(GtkRange *range, gpointer data) {
    KMEANS_CLUSTERS = (int)gtk_range_get_value(range);
    
    // Only update if kmeans is selected
    gchar *selected_algorithm = gtk_combo_box_text_get_active_text(GTK_COMBO_BOX_TEXT(algorithm_combo));
    if (selected_algorithm != NULL && strcmp(selected_algorithm, "K-Means") == 0) {
        update_kmeans_segmentation();
    }
    g_free(selected_algorithm);
}

// Update backtracking segmentation with current threshold
static void update_backtracking_segmentation() {
    if (filename == NULL || input_image.empty()) {
        return;
    }

    try {
        auto start_time = chrono::high_resolution_clock::now();
        
        Mat processed_image = backtrackingSegmentation(input_image);
        
        auto end_time = chrono::high_resolution_clock::now();
        double elapsed_time = chrono::duration<double, milli>(end_time - start_time).count();

        if (!processed_image.empty()) {
            string temp_filename = string(filename) + "_processed.jpg";
            imwrite(temp_filename, processed_image);

            GdkPixbuf *pixbuf = gdk_pixbuf_new_from_file_at_scale(temp_filename.c_str(), 
                                                             400, 400, 
                                                             TRUE, 
                                                             NULL);
            if (pixbuf) {
                gtk_image_set_from_pixbuf(GTK_IMAGE(processed_image_view), pixbuf);
                g_object_unref(pixbuf);
                
                gtk_label_set_text(GTK_LABEL(status_label), 
                    g_strdup_printf("Processing Time: %.2f ms", elapsed_time));
                
                gtk_label_set_text(GTK_LABEL(threshold_label), 
                    g_strdup_printf("Parameters:\nBacktracking threshold: %d", 
                    BACKTRACKING_THRESHOLD));
            }
        }
    } catch (const cv::Exception& e) {
        gtk_label_set_text(GTK_LABEL(status_label), g_strdup_printf("Error: %s", e.what()));
    }
}

// Update improved backtracking segmentation with current threshold
static void update_backtracking_improved_segmentation() {
    if (filename == NULL || input_image.empty()) {
        return;
    }

    try {
        auto start_time = chrono::high_resolution_clock::now();
        
        Mat processed_image = backtrackingSegmentationImproved(input_image);
        
        auto end_time = chrono::high_resolution_clock::now();
        double elapsed_time = chrono::duration<double, milli>(end_time - start_time).count();

        if (!processed_image.empty()) {
            string temp_filename = string(filename) + "_processed.jpg";
            imwrite(temp_filename, processed_image);

            GdkPixbuf *pixbuf = gdk_pixbuf_new_from_file_at_scale(temp_filename.c_str(), 
                                                             400, 400, 
                                                             TRUE, 
                                                             NULL);
            if (pixbuf) {
                gtk_image_set_from_pixbuf(GTK_IMAGE(processed_image_view), pixbuf);
                g_object_unref(pixbuf);
                
                gtk_label_set_text(GTK_LABEL(status_label), 
                    g_strdup_printf("Processing Time: %.2f ms", elapsed_time));
                
                gtk_label_set_text(GTK_LABEL(threshold_label), 
                    g_strdup_printf("Parameters:\nBacktracking threshold: %d\nBilateral filter: sigma=75", 
                    BACKTRACKING_THRESHOLD));
            }
        }
    } catch (const cv::Exception& e) {
        gtk_label_set_text(GTK_LABEL(status_label), g_strdup_printf("Error: %s", e.what()));
    }
}

// Update kmeans segmentation with current number of clusters
static void update_kmeans_segmentation() {
    if (filename == NULL || input_image.empty()) {
        return;
    }

    try {
        auto start_time = chrono::high_resolution_clock::now();
        
        Mat processed_image = kMeansSegmentation(input_image, KMEANS_CLUSTERS);
        
        auto end_time = chrono::high_resolution_clock::now();
        double elapsed_time = chrono::duration<double, milli>(end_time - start_time).count();

        if (!processed_image.empty()) {
            string temp_filename = string(filename) + "_processed.jpg";
            imwrite(temp_filename, processed_image);

            GdkPixbuf *pixbuf = gdk_pixbuf_new_from_file_at_scale(temp_filename.c_str(), 
                                                             400, 400, 
                                                             TRUE, 
                                                             NULL);
            if (pixbuf) {
                gtk_image_set_from_pixbuf(GTK_IMAGE(processed_image_view), pixbuf);
                g_object_unref(pixbuf);
                
                gtk_label_set_text(GTK_LABEL(status_label), 
                    g_strdup_printf("Processing Time: %.2f ms", elapsed_time));
                
                gtk_label_set_text(GTK_LABEL(threshold_label), 
                    g_strdup_printf("Parameters:\nNumber of clusters: %d\nMax iterations: %d\nEpsilon: %.1f", 
                    KMEANS_CLUSTERS, KMEANS_MAX_ITER, KMEANS_EPSILON));
            }
        }
    } catch (const cv::Exception& e) {
        gtk_label_set_text(GTK_LABEL(status_label), g_strdup_printf("Error: %s", e.what()));
    }
}

// Callback for algorithm selection change
static void on_algorithm_changed(GtkComboBox *widget, gpointer data) {
    gchar *selected_algorithm = gtk_combo_box_text_get_active_text(GTK_COMBO_BOX_TEXT(widget));
    
    // Show/hide appropriate sliders based on algorithm selection
    if (selected_algorithm != NULL) {
        if (strcmp(selected_algorithm, "Backtracking") == 0 || 
            strcmp(selected_algorithm, "Backtracking (8-Dir)") == 0 ||
            strcmp(selected_algorithm, "Backtracking Improved") == 0 ||
            strcmp(selected_algorithm, "Backtracking Edge Enhanced") == 0) {
            gtk_widget_show_all(threshold_slider_box);
            gtk_widget_hide(kmeans_slider_box);
        } else if (strcmp(selected_algorithm, "K-Means") == 0) {
            gtk_widget_hide(threshold_slider_box);
            gtk_widget_show_all(kmeans_slider_box);
        } else {
            gtk_widget_hide(threshold_slider_box);
            gtk_widget_hide(kmeans_slider_box);
        }
    }
    
    g_free(selected_algorithm);
}

// Apply the selected algorithm to the image
static void apply_algorithm(GtkWidget *widget, gpointer data) {
    if (filename == NULL || input_image.empty()) {
        gtk_label_set_text(GTK_LABEL(status_label), "Please select an image first");
        return;
    }

    // Get the selected algorithm
    gchar *selected_algorithm = gtk_combo_box_text_get_active_text(GTK_COMBO_BOX_TEXT(algorithm_combo));
    if (selected_algorithm == NULL) {
        gtk_label_set_text(GTK_LABEL(status_label), "Please select an algorithm");
        return;
    }

    // Close any existing histogram window
    destroyWindow("Otsu Threshold Histogram");

    gtk_label_set_text(GTK_LABEL(status_label), g_strdup_printf("Applying %s...", selected_algorithm));
    gtk_label_set_text(GTK_LABEL(info_label), ""); // Clear previous info
    gtk_label_set_text(GTK_LABEL(threshold_label), ""); // Clear previous threshold info
    
    Mat processed_image;
    string algorithm_info = "";
    string threshold_info = "";
    
    try {
        // Start measuring time
        auto start_time = chrono::high_resolution_clock::now();
        
        if (strcmp(selected_algorithm, "Active Contours") == 0) {
            processed_image = activeContoursSegmentation(input_image);
            algorithm_info = "Active Contours: Using edge detection and contour evolution";
            threshold_info = g_strdup_printf("Parameters:\n"
                                          "Iterations: %d\n"
                                          "Alpha (Elasticity): %.2f\n"
                                          "Beta (Curvature): %.2f\n"
                                          "Gamma (External Energy): %.2f",
                                          ACTIVE_CONTOURS_ITERATIONS,
                                          ACTIVE_CONTOURS_ALPHA,
                                          ACTIVE_CONTOURS_BETA,
                                          ACTIVE_CONTOURS_GAMMA);
        } else if (strcmp(selected_algorithm, "K-Means") == 0) {
            processed_image = kMeansSegmentation(input_image, KMEANS_CLUSTERS);
            algorithm_info = "K-Means: Clustering based segmentation";
            threshold_info = g_strdup_printf("Parameters:\n"
                                          "Clusters: %d\n"
                                          "Max Iterations: %d\n"
                                          "Epsilon: %.1f",
                                          KMEANS_CLUSTERS,
                                          KMEANS_MAX_ITER,
                                          KMEANS_EPSILON);
        } else if (strcmp(selected_algorithm, "Otsu Thresholding") == 0) {
            double otsuThreshold;
            processed_image = otsuSegmentation(input_image, otsuThreshold);
            algorithm_info = "Otsu: Automatic threshold selection";
            threshold_info = g_strdup_printf("Parameters:\nComputed threshold: %.1f", otsuThreshold);
        } else if (strcmp(selected_algorithm, "Backtracking") == 0) {
            processed_image = backtrackingSegmentation(input_image);
            algorithm_info = "Backtracking: 4-directional region-based segmentation";
            threshold_info = g_strdup_printf("Parameters:\nThreshold: %d", BACKTRACKING_THRESHOLD);
        } else if (strcmp(selected_algorithm, "Backtracking (8-Dir)") == 0) {
            processed_image = backtrackingSegmentation8Dir(input_image);
            algorithm_info = "Backtracking: 8-directional region-based segmentation with noise reduction";
            threshold_info = g_strdup_printf("Parameters:\nThreshold: %d\nGaussian blur: 3x3", BACKTRACKING_THRESHOLD);
        } else if (strcmp(selected_algorithm, "Backtracking Improved") == 0) {
            processed_image = backtrackingSegmentationImproved(input_image);
            algorithm_info = "Backtracking Improved: Region-based segmentation with bilateral filter";
            threshold_info = g_strdup_printf("Parameters:\nThreshold: %d\nBilateral filter: sigma=75", BACKTRACKING_THRESHOLD);
        } else if (strcmp(selected_algorithm, "Backtracking Edge Enhanced") == 0) {
            processed_image = backtrackingEdgeEnhancementSegmentation(input_image);
            algorithm_info = "Backtracking Edge Enhanced: Region-based segmentation with edge enhancement";
            threshold_info = g_strdup_printf("Parameters:\nThreshold: %d", BACKTRACKING_THRESHOLD);
        } else if (strcmp(selected_algorithm, "Watershed") == 0) {
            processed_image = watershedSegmentation(input_image);
            algorithm_info = "Watershed: Morphological segmentation";
            threshold_info = g_strdup_printf("Parameters:\n"
                                          "Morphological kernel size: %d",
                                          WATERSHED_MORPH_SIZE);
        } else if (strcmp(selected_algorithm, "Graph Cut") == 0) {
            processed_image = graphCutSegmentation(input_image);
            algorithm_info = "Graph Cut: Using GrabCut algorithm";
            threshold_info = g_strdup_printf("Parameters:\n"
                                          "GrabCut iterations: %d",
                                          GRAPH_CUT_ITERATIONS);
        } else if (strcmp(selected_algorithm, "Region Growing") == 0) {
            Point seed(input_image.cols / 2, input_image.rows / 2);
            processed_image = regionGrowingSegmentation(input_image, seed, REGION_GROWING_THRESHOLD);
            algorithm_info = "Region Growing: Seed-based segmentation";
            threshold_info = g_strdup_printf("Parameters:\n"
                                          "Intensity threshold: %d\n"
                                          "Seed point: center of image",
                                          REGION_GROWING_THRESHOLD);
        } else {
            gtk_label_set_text(GTK_LABEL(status_label), "Unknown algorithm selected");
            g_free(selected_algorithm);
            return;
        }
        
        // Stop measuring time
        auto end_time = chrono::high_resolution_clock::now();
        double elapsed_time = chrono::duration<double, milli>(end_time - start_time).count();

        g_free(selected_algorithm);

        if (processed_image.empty()) {
            gtk_label_set_text(GTK_LABEL(status_label), "Failed to process image");
            return;
        }

        // Create a temporary file to save the processed image
        string temp_filename = string(filename) + "_processed.jpg";
        imwrite(temp_filename, processed_image);

        // Display the processed image
        GdkPixbuf *pixbuf = gdk_pixbuf_new_from_file_at_scale(temp_filename.c_str(), 
                                                         400, 400, 
                                                         TRUE, 
                                                         NULL);
        if (pixbuf) {
            gtk_image_set_from_pixbuf(GTK_IMAGE(processed_image_view), pixbuf);
            g_object_unref(pixbuf);
            
            // Update status with larger time display
            gtk_label_set_text(GTK_LABEL(status_label), 
                g_strdup_printf("Processing Time: %.2f ms", elapsed_time));
            
            // Update info label with algorithm details
            gtk_label_set_text(GTK_LABEL(info_label), algorithm_info.c_str());

            // Update threshold label with parameter details
            gtk_label_set_text(GTK_LABEL(threshold_label), threshold_info.c_str());
        } else {
            gtk_label_set_text(GTK_LABEL(status_label), "Failed to display processed image");
        }
    } catch (const cv::Exception& e) {
        gtk_label_set_text(GTK_LABEL(status_label), g_strdup_printf("Error: %s", e.what()));
        g_free(selected_algorithm);
        return;
    }
}

// Open file dialog to select an image
static void select_image(GtkWidget *widget, gpointer data) {
    GtkWidget *dialog;
    GtkFileChooserAction action = GTK_FILE_CHOOSER_ACTION_OPEN;
    gint res;

    dialog = gtk_file_chooser_dialog_new("Open Image",
                                        GTK_WINDOW(window),
                                        action,
                                        "_Cancel",
                                        GTK_RESPONSE_CANCEL,
                                        "_Open",
                                        GTK_RESPONSE_ACCEPT,
                                        NULL);

    // Add filters for image files
    GtkFileFilter *filter = gtk_file_filter_new();
    gtk_file_filter_set_name(filter, "Image Files");
    gtk_file_filter_add_mime_type(filter, "image/jpeg");
    gtk_file_filter_add_mime_type(filter, "image/png");
    gtk_file_filter_add_mime_type(filter, "image/bmp");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);

    res = gtk_dialog_run(GTK_DIALOG(dialog));
    if (res == GTK_RESPONSE_ACCEPT) {
        GtkFileChooser *chooser = GTK_FILE_CHOOSER(dialog);
        if (filename != NULL) {
            g_free(filename);
        }
        filename = gtk_file_chooser_get_filename(chooser);
        
        // Load the image for processing
        input_image = imread(filename, IMREAD_COLOR);
        if (input_image.empty()) {
            gtk_label_set_text(GTK_LABEL(status_label), "Failed to load image");
            gtk_widget_destroy(dialog);
            return;
        }
        
        // Display the selected image
        GdkPixbuf *pixbuf = gdk_pixbuf_new_from_file_at_scale(filename, 
                                                             400, 400, 
                                                             TRUE, 
                                                             NULL);
        if (pixbuf) {
            gtk_image_set_from_pixbuf(GTK_IMAGE(original_image_view), pixbuf);
            g_object_unref(pixbuf);
            gtk_label_set_text(GTK_LABEL(status_label), "Image loaded successfully");
            gtk_widget_set_sensitive(apply_button, TRUE);
        } else {
            gtk_label_set_text(GTK_LABEL(status_label), "Failed to display image");
        }
    }

    gtk_widget_destroy(dialog);
}

// Initialize the GTK application
static void activate(GtkApplication *app, gpointer user_data) {
    // Create the main window
    window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(window), "Image Segmentation App");
    gtk_window_set_default_size(GTK_WINDOW(window), 900, 600);
    gtk_container_set_border_width(GTK_CONTAINER(window), 10);

    // Create a vertical box to organize widgets
    GtkWidget *main_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_container_add(GTK_CONTAINER(window), main_box);

    // Create a horizontal box for controls
    GtkWidget *control_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    gtk_box_pack_start(GTK_BOX(main_box), control_box, FALSE, FALSE, 0);

    // Create a button to select an image
    GtkWidget *select_button = gtk_button_new_with_label("Select Image");
    g_signal_connect(select_button, "clicked", G_CALLBACK(select_image), NULL);
    gtk_box_pack_start(GTK_BOX(control_box), select_button, FALSE, FALSE, 0);

    // Create a combo box for algorithm selection
    algorithm_combo = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(algorithm_combo), "Active Contours");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(algorithm_combo), "K-Means");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(algorithm_combo), "Otsu Thresholding");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(algorithm_combo), "Backtracking");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(algorithm_combo), "Backtracking (8-Dir)");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(algorithm_combo), "Backtracking Improved");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(algorithm_combo), "Backtracking Edge Enhanced");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(algorithm_combo), "Watershed");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(algorithm_combo), "Graph Cut");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(algorithm_combo), "Region Growing");
    gtk_combo_box_set_active(GTK_COMBO_BOX(algorithm_combo), 0);
    g_signal_connect(algorithm_combo, "changed", G_CALLBACK(on_algorithm_changed), NULL);
    gtk_box_pack_start(GTK_BOX(control_box), algorithm_combo, FALSE, FALSE, 0);

    // Create a button to apply the selected algorithm
    apply_button = gtk_button_new_with_label("Apply Algorithm");
    g_signal_connect(apply_button, "clicked", G_CALLBACK(apply_algorithm), NULL);
    gtk_box_pack_start(GTK_BOX(control_box), apply_button, FALSE, FALSE, 0);
    gtk_widget_set_sensitive(apply_button, FALSE); // Disable until image is loaded

    // Create backtracking threshold slider box
    threshold_slider_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    gtk_box_pack_start(GTK_BOX(control_box), threshold_slider_box, TRUE, TRUE, 0);

    // Create backtracking threshold slider
    GtkWidget *threshold_label = gtk_label_new("Threshold:");
    gtk_box_pack_start(GTK_BOX(threshold_slider_box), threshold_label, FALSE, FALSE, 0);

    threshold_slider = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0, 255, 1);
    gtk_range_set_value(GTK_RANGE(threshold_slider), BACKTRACKING_THRESHOLD);
    gtk_widget_set_size_request(threshold_slider, 200, -1);
    g_signal_connect(threshold_slider, "value-changed", G_CALLBACK(on_threshold_changed), NULL);
    gtk_box_pack_start(GTK_BOX(threshold_slider_box), threshold_slider, TRUE, TRUE, 0);

    // Create kmeans clusters slider box
    kmeans_slider_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    gtk_box_pack_start(GTK_BOX(control_box), kmeans_slider_box, TRUE, TRUE, 0);

    // Create kmeans clusters slider
    GtkWidget *kmeans_label = gtk_label_new("Clusters:");
    gtk_box_pack_start(GTK_BOX(kmeans_slider_box), kmeans_label, FALSE, FALSE, 0);

    kmeans_slider = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 2, 8, 1);
    gtk_range_set_value(GTK_RANGE(kmeans_slider), KMEANS_CLUSTERS);
    gtk_widget_set_size_request(kmeans_slider, 200, -1);
    g_signal_connect(kmeans_slider, "value-changed", G_CALLBACK(on_kmeans_changed), NULL);
    gtk_box_pack_start(GTK_BOX(kmeans_slider_box), kmeans_slider, TRUE, TRUE, 0);

    // Hide both slider boxes initially
    gtk_widget_hide(threshold_slider_box);
    gtk_widget_hide(kmeans_slider_box);

    // Create a horizontal box for images
    GtkWidget *image_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    gtk_box_pack_start(GTK_BOX(main_box), image_box, TRUE, TRUE, 0);

    // Create frames for original and processed images
    GtkWidget *original_frame = gtk_frame_new("Original Image");
    GtkWidget *processed_frame = gtk_frame_new("Processed Image");
    
    gtk_box_pack_start(GTK_BOX(image_box), original_frame, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(image_box), processed_frame, TRUE, TRUE, 0);

    // Create image widgets
    original_image_view = gtk_image_new();
    processed_image_view = gtk_image_new();
    
    // Add images to frames
    gtk_container_add(GTK_CONTAINER(original_frame), original_image_view);
    gtk_container_add(GTK_CONTAINER(processed_frame), processed_image_view);

    // Create status label with larger text
    status_label = gtk_label_new("Ready");
    PangoAttrList *attr_list = pango_attr_list_new();
    PangoAttribute *attr = pango_attr_scale_new(1.5); // Make text 1.5 times larger
    pango_attr_list_insert(attr_list, attr);
    gtk_label_set_attributes(GTK_LABEL(status_label), attr_list);
    pango_attr_list_unref(attr_list);
    gtk_box_pack_start(GTK_BOX(main_box), status_label, FALSE, FALSE, 0);

    // Create info label for algorithm details
    info_label = gtk_label_new("");
    gtk_box_pack_start(GTK_BOX(main_box), info_label, FALSE, FALSE, 0);

    // Create threshold label
    threshold_label = gtk_label_new("");
    gtk_box_pack_start(GTK_BOX(main_box), threshold_label, FALSE, FALSE, 0);

    // Show all widgets
    gtk_widget_show_all(window);
}

// Main function
int main(int argc, char **argv) {
    GtkApplication *app;
    int status;

    app = gtk_application_new("org.gtk.example", G_APPLICATION_DEFAULT_FLAGS);
    g_signal_connect(app, "activate", G_CALLBACK(activate), NULL);
    status = g_application_run(G_APPLICATION(app), argc, argv);
    g_object_unref(app);

    return status;
}

// Active Contours Segmentation Implementation
Mat activeContoursSegmentation(const Mat& image) {
    Mat gray;
    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    // Step 1: Edge Detection
    Mat edges;
    Canny(gray, edges, 50, 150);

    // Step 2: Contour Finding
    vector<vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        throw cv::Exception(0, "No contours found in the image", "activeContoursSegmentation", __FILE__, __LINE__);
    }

    // Step 3: Select Largest Contour
    int largest_contour_idx = 0;
    double largest_area = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > largest_area) {
            largest_area = area;
            largest_contour_idx = i;
        }
    }

    // Step 4: Snake Evolution
    vector<Point> snake = contours[largest_contour_idx];
    for (int iter = 0; iter < ACTIVE_CONTOURS_ITERATIONS; iter++) {
        for (size_t i = 0; i < snake.size(); i++) {
            int prev = (i == 0) ? snake.size() - 1 : i - 1;
            int next = (i == snake.size() - 1) ? 0 : i + 1;
            Point newPoint = (1 - ACTIVE_CONTOURS_ALPHA) * snake[i] + ACTIVE_CONTOURS_ALPHA * (snake[prev] + snake[next]) / 2;
            if (newPoint.x >= 0 && newPoint.y >= 0 && newPoint.x < edges.cols && newPoint.y < edges.rows) {
                if (edges.at<uchar>(newPoint) > 0) {
                    newPoint = snake[i] + ACTIVE_CONTOURS_GAMMA * (newPoint - snake[i]);
                }
            }
            snake[i] = newPoint;
        }
    }

    // Step 5: Visualization
    Mat result = image.clone();
    for (size_t i = 0; i < snake.size(); i++) {
        int next = (i == snake.size() - 1) ? 0 : i + 1;
        line(result, snake[i], snake[next], Scalar(0, 255, 0), 2);
    }

    return result;
}

// K-Means Segmentation Implementation
Mat kMeansSegmentation(const Mat& image, int clusters) {
    // Convert to grayscale if not already
    Mat gray;
    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    Mat data;
    gray.convertTo(data, CV_32F); 
    data = data.reshape(1, gray.total()); 

    // Apply k-means clustering
    Mat labels, centers;
    kmeans(data, clusters, labels, TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 1.0), 3, KMEANS_RANDOM_CENTERS, centers);

    Mat segmented(gray.size(), CV_8U);
    for (int i = 0; i < data.rows; i++) {
        segmented.at<uchar>(i / gray.cols, i % gray.cols) = (uchar)centers.at<float>(labels.at<int>(i, 0), 0);
    }


    Mat colored;
    applyColorMap(segmented, colored, COLORMAP_JET);

    return colored;
}

// Otsu Segmentation Implementation
Mat otsuSegmentation(const Mat& image, double& otsuThreshold) {
    Mat gray;
    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    // Apply Otsu's thresholding
    Mat segmented;
    otsuThreshold = threshold(gray, segmented, 0, 255, THRESH_BINARY | THRESH_OTSU);
    
    // Calculate and display histogram in a separate window
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    Mat hist;
    calcHist(&gray, 1, 0, Mat(), hist, 1, &histSize, &histRange);

    // Create histogram visualization
    Mat histImage(200, 512, CV_8UC3, Scalar(255, 255, 255));
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX);
    
    // Draw histogram
    for(int i = 0; i < histSize; i++) {
        line(histImage,
             Point(i*2, histImage.rows),
             Point(i*2, histImage.rows - cvRound(hist.at<float>(i))),
             Scalar(100, 100, 100),
             2);
    }

    // Draw threshold line
    line(histImage,
         Point(cvRound(otsuThreshold)*2, 0),
         Point(cvRound(otsuThreshold)*2, histImage.rows),
         Scalar(0, 0, 255),
         2);
    
    // Add text for threshold value
    putText(histImage,
            format("Threshold: %.1f", otsuThreshold),
            Point(10, 30),
            FONT_HERSHEY_SIMPLEX,
            0.7,
            Scalar(0, 0, 0),
            2);

    // Show histogram in separate window
    imshow("Otsu Threshold Histogram", histImage);
    
    // Convert segmented image to color for main display
    Mat colored;
    cvtColor(segmented, colored, COLOR_GRAY2BGR);
    return colored;
}

// Basic Backtracking Segmentation Implementation
Mat backtrackingSegmentation(const Mat& image) {
    Mat gray;
    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    // Define threshold value
    int threshValue = BACKTRACKING_THRESHOLD;

    // Create a matrix for segmentation
    Mat binary;
    threshold(gray, binary, threshValue, 255, THRESH_BINARY);
    
    // Create a copy for the result
    Mat segmented = binary.clone();
    
    // Choose a starting point for segmentation (center of image)
    int startX = gray.cols / 2;
    int startY = gray.rows / 2;
    int oldColor = binary.at<uchar>(startY, startX);
    int newColor = 128; // Mid-gray for marking the segmented region
    
    // Use a non-recursive approach to avoid stack overflow
    Mat visited = Mat::zeros(binary.size(), CV_8UC1);
    queue<Point> q;
    q.push(Point(startX, startY));
    visited.at<uchar>(startY, startX) = 1;
    
    // Define 4-directional connectivity 
    const int dx[] = {0, 0, 1, -1};  // East, West, North, South
    const int dy[] = {1, -1, 0, 0};  // No diagonals
    const int NUM_DIRECTIONS = 4;
    
    while (!q.empty()) {
        Point p = q.front();
        q.pop();
        
        // If the pixel matches the old color, mark it with the new color
        if (binary.at<uchar>(p.y, p.x) == oldColor) {
            segmented.at<uchar>(p.y, p.x) = newColor;
            
            // Check only 4-connected neighbors
            for (int i = 0; i < NUM_DIRECTIONS; i++) {
                int nx = p.x + dx[i];
                int ny = p.y + dy[i];
                
                if (nx >= 0 && ny >= 0 && nx < binary.cols && ny < binary.rows && 
                    !visited.at<uchar>(ny, nx)) {
                    visited.at<uchar>(ny, nx) = 1;
                    q.push(Point(nx, ny));
                }
            }
        }
    }
    
    // Apply color map for better visualization
    Mat colored;
    applyColorMap(segmented, colored, COLORMAP_JET);
    
    return colored;
}

// Improved Backtracking Segmentation Implementation
Mat backtrackingSegmentationImproved(const Mat& image) {
    // Convert to grayscale if not already
    Mat gray;
    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    // Apply bilateral filter to reduce noise but keep edges
    Mat smooth;
    bilateralFilter(gray, smooth, 9, 75, 75);

    // Define threshold value
    int threshValue = BACKTRACKING_THRESHOLD;

    // Threshold the smoothed image
    Mat binary;
    threshold(smooth, binary, threshValue, 255, THRESH_BINARY);

    // Copy binary image for segmentation
    Mat segmented = binary.clone();

    // Starting point: center of image
    int startX = gray.cols / 2;
    int startY = gray.rows / 2;
    int oldColor = binary.at<uchar>(startY, startX);
    int newColor = 128;

    // Non-recursive flood-fill using BFS
    Mat visited = Mat::zeros(binary.size(), CV_8UC1);
    queue<Point> q;
    q.push(Point(startX, startY));
    visited.at<uchar>(startY, startX) = 1;

    while (!q.empty()) {
        Point p = q.front();
        q.pop();

        if (binary.at<uchar>(p.y, p.x) == oldColor) {
            segmented.at<uchar>(p.y, p.x) = newColor;

            int dx[] = {1, -1, 0, 0};
            int dy[] = {0, 0, 1, -1};

            for (int i = 0; i < 4; i++) {
                int nx = p.x + dx[i];
                int ny = p.y + dy[i];

                if (nx >= 0 && ny >= 0 && nx < binary.cols && ny < binary.rows &&
                    !visited.at<uchar>(ny, nx)) {
                    visited.at<uchar>(ny, nx) = 1;
                    q.push(Point(nx, ny));
                }
            }
        }
    }

    // Morphological post-processing to refine regions
    Mat morph;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(segmented, morph, MORPH_CLOSE, kernel);

    // Apply color map for visualization
    Mat colored;
    applyColorMap(morph, colored, COLORMAP_JET);

    return colored;
}

// Watershed Segmentation Implementation
Mat watershedSegmentation(const Mat& image) {
    // Ensure image is in color
    Mat colorImage;
    if (image.channels() == 1) {
        cvtColor(image, colorImage, COLOR_GRAY2BGR);
    } else {
        colorImage = image.clone();
    }

    // Convert to grayscale for processing
    Mat gray;
    cvtColor(colorImage, gray, COLOR_BGR2GRAY);

    // Apply Otsu's thresholding to create a binary image
    Mat binary;
    threshold(gray, binary, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);

    // Remove noise using morphological operations
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat sureBg;
    dilate(binary, sureBg, kernel, Point(-1, -1), 3);

    // Find sure foreground area using distance transform
    Mat distTransform;
    distanceTransform(binary, distTransform, DIST_L2, 5);
    normalize(distTransform, distTransform, 0, 1.0, NORM_MINMAX);
    Mat sureFg;
    threshold(distTransform, sureFg, 0.5, 1.0, THRESH_BINARY);
    
    // Convert sure foreground to 8-bit image
    sureFg.convertTo(sureFg, CV_8U);

    // Find unknown region
    Mat unknown;
    subtract(sureBg, sureFg, unknown);

    // Label markers
    Mat markers;
    connectedComponents(sureFg, markers);

    // Add 1 to all markers so that the unknown region is marked as 0
    markers = markers + 1;

    // Mark the unknown region with 0
    markers.setTo(0, unknown == 255);

    // Apply Watershed algorithm
    watershed(colorImage, markers);

    // Create an output image
    Mat segmented = Mat::zeros(colorImage.size(), CV_8UC3);
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            if (markers.at<int>(i, j) == -1) {
                segmented.at<Vec3b>(i, j) = Vec3b(0, 0, 255); // Mark boundaries in red
            } else if (markers.at<int>(i, j) > 1) {
                segmented.at<Vec3b>(i, j) = colorImage.at<Vec3b>(i, j); // Keep original color
            }
        }
    }

    return segmented;
}

// Graph Cut Segmentation Implementation
Mat graphCutSegmentation(const Mat& image) {
    // Ensure image is in color
    Mat colorImage;
    if (image.channels() == 1) {
        cvtColor(image, colorImage, COLOR_GRAY2BGR);
    } else {
        colorImage = image.clone();
    }

    // Define a rectangle around the object of interest (center of the image)
    int margin = min(colorImage.cols, colorImage.rows) / 4;
    Rect rectangle(margin, margin, colorImage.cols - 2*margin, colorImage.rows - 2*margin);

    // Initialize mask
    Mat mask(colorImage.size(), CV_8UC1, Scalar(cv::GC_BGD)); // Default: Background

    // Initialize background and foreground models
    Mat bgModel, fgModel;

    // Apply GrabCut (Graph Cut)
    grabCut(colorImage, mask, rectangle, bgModel, fgModel, 5, cv::GC_INIT_WITH_RECT);

    // Convert mask to binary: Foreground pixels are marked
    Mat segmented;
    compare(mask, cv::GC_PR_FGD, segmented, CMP_EQ);
    
    // Convert to 3-channel image for visualization
    Mat output(colorImage.size(), CV_8UC3, Scalar(0, 0, 0));
    colorImage.copyTo(output, segmented);

    return output;
}

// Region Growing Segmentation Implementation
Mat regionGrowingSegmentation(const Mat& image, Point seed, int threshold) {
    // Convert to grayscale if not already
    Mat gray;
    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    Mat segmented = Mat::zeros(gray.size(), CV_8UC1); 
    Mat visited = Mat::zeros(gray.size(), CV_8UC1);   // Track visited pixels
    queue<Point> pixelQueue; 

    int seedIntensity = gray.at<uchar>(seed.y, seed.x);
    pixelQueue.push(seed);
    visited.at<uchar>(seed.y, seed.x) = 1;


    int dx[] = {0, 0, -1, 1};
    int dy[] = {-1, 1, 0, 0};

    while (!pixelQueue.empty()) {
        Point p = pixelQueue.front();
        pixelQueue.pop();
        segmented.at<uchar>(p.y, p.x) = 255; 

        // Check neighbors
        for (int i = 0; i < 4; i++) {
            int nx = p.x + dx[i];
            int ny = p.y + dy[i];

            // Boundary check
            if (nx >= 0 && ny >= 0 && nx < gray.cols && ny < gray.rows) {
                int neighborIntensity = gray.at<uchar>(ny, nx);
                if (!visited.at<uchar>(ny, nx) && abs(neighborIntensity - seedIntensity) < threshold) {
                    visited.at<uchar>(ny, nx) = 1;
                    pixelQueue.push(Point(nx, ny));
                }
            }
        }
    }
    
    Mat colored;
    applyColorMap(segmented, colored, COLORMAP_JET);
    
    return colored;
}

// Advanced Backtracking with Edge Enhancement Implementation
Mat backtrackingEdgeEnhancementSegmentation(const Mat& image) {
    Mat gray;
    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    // Step 1: Advanced Pre-processing
    Mat denoised, enhanced;
    bilateralFilter(gray, denoised, 9, 75, 75);
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    clahe->apply(denoised, enhanced);

    // Step 2: Multi-scale Edge Detection
    Mat gradX, gradY, gradMag;
    Sobel(enhanced, gradX, CV_32F, 1, 0, 3);
    Sobel(enhanced, gradY, CV_32F, 0, 1, 3);
    magnitude(gradX, gradY, gradMag);
    normalize(gradMag, gradMag, 0, 255, NORM_MINMAX);
    gradMag.convertTo(gradMag, CV_8U);

    // Step 3: Initial Segmentation
    Mat binary;
    adaptiveThreshold(enhanced, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 21, 5);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binary, binary, MORPH_CLOSE, kernel);

    // Step 4: Region Growing with Smart Backtracking
    Mat segmented = Mat::zeros(gray.size(), CV_8UC1);
    Mat visited = Mat::zeros(gray.size(), CV_8UC1);
    vector<Point> seeds;
    int gridSize = 3;
    for (int i = 1; i <= gridSize; i++) {
        for (int j = 1; j <= gridSize; j++) {
            seeds.push_back(Point((gray.cols * i) / (gridSize + 1), (gray.rows * j) / (gridSize + 1)));
        }
    }

    // Process each seed point with region growing
    for (const Point& seed : seeds) {
        if (visited.at<uchar>(seed.y, seed.x)) continue;

        queue<Point> q;
        q.push(seed);
        visited.at<uchar>(seed.y, seed.x) = 1;

        // Reference values for region growing
        int refIntensity = enhanced.at<uchar>(seed.y, seed.x);
        double refGradient = gradMag.at<uchar>(seed.y, seed.x);

        // 8-connectivity for region growing
        int dx[] = {1, -1, 0, 0, 1, -1, 1, -1};
        int dy[] = {0, 0, 1, -1, 1, -1, -1, 1};

        while (!q.empty()) {
            Point p = q.front();
            q.pop();

            // Check if this pixel should be included in the segment
            bool isValidRegion = binary.at<uchar>(p.y, p.x) > 0;
            if (isValidRegion) {
                segmented.at<uchar>(p.y, p.x) = 255;

                // Check neighbors
                for (int i = 0; i < 8; i++) {
                    int nx = p.x + dx[i];
                    int ny = p.y + dy[i];

                    if (nx >= 0 && ny >= 0 && nx < gray.cols && ny < gray.rows && 
                        !visited.at<uchar>(ny, nx)) {
                        
                        // Multi-criteria region growing
                        int intensityDiff = abs(enhanced.at<uchar>(ny, nx) - refIntensity);
                        double gradientDiff = abs(gradMag.at<uchar>(ny, nx) - refGradient);
                        
                        bool isValidPixel = 
                            // Intensity similarity
                            intensityDiff < BACKTRACKING_THRESHOLD &&
                            // Gradient continuity
                            gradientDiff < BACKTRACKING_THRESHOLD * 0.5 &&
                            // Edge strength consideration
                            gradMag.at<uchar>(ny, nx) < BACKTRACKING_THRESHOLD * 1.5;

                        if (isValidPixel) {
                            visited.at<uchar>(ny, nx) = 1;
                            q.push(Point(nx, ny));
                        }
                    }
                }
            }
        }
    }

    // Step 5: Post-processing and Visualization
    Mat result = image.clone();
    
    // Draw contours with different colors based on confidence
    vector<vector<Point>> contours;
    findContours(segmented, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Filter contours based on area and shape
    vector<vector<Point>> filteredContours;
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        double perimeter = arcLength(contour, true);
        double circularity = 4 * M_PI * area / (perimeter * perimeter);
        
        // Filter based on area and shape (doors are typically rectangular)
        if (area > 1000 && circularity < 0.8) {
            // Approximate contour to reduce noise
            vector<Point> approx;
            approxPolyDP(contour, approx, 0.02 * perimeter, true);
            
            // Add if the shape has 4-8 vertices (door-like shape)
            if (approx.size() >= 4 && approx.size() <= 8) {
                filteredContours.push_back(approx);
            }
        }
    }

    // Draw contours with different colors based on confidence
    for (const auto& contour : filteredContours) {
        // Calculate confidence based on contour properties
        double area = contourArea(contour);
        double maxArea = gray.rows * gray.cols;
        double confidence = min(area / maxArea * 4, 1.0); // Normalize confidence

        // Use color gradient based on confidence (green to yellow)
        Scalar color(0, 255, static_cast<int>(255 * (1 - confidence)));
        drawContours(result, vector<vector<Point>>{contour}, -1, color, 2);
    }

    // Create semi-transparent overlay
    Mat overlay = Mat::zeros(result.size(), result.type());
    for (const auto& contour : filteredContours) {
        fillPoly(overlay, vector<vector<Point>>{contour}, Scalar(0, 0, 255));
    }
    
    // Combine with transparency
    addWeighted(result, 0.7, overlay, 0.3, 0, result);

    // Add confidence visualization
    for (size_t i = 0; i < filteredContours.size(); i++) {
        // Calculate centroid
        Moments mu = moments(filteredContours[i]);
        Point centroid(mu.m10/mu.m00, mu.m01/mu.m00);
        
        // Calculate and display confidence score
        double area = contourArea(filteredContours[i]);
        double maxArea = gray.rows * gray.cols;
        int confidence = static_cast<int>(min(area / maxArea * 400, 100.0));
        
        putText(result, format("Conf: %d%%", confidence), 
                Point(centroid.x - 40, centroid.y),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
    }

    return result;
}

// 8-Directional Backtracking Segmentation Implementation
Mat backtrackingSegmentation8Dir(const Mat& image) {
    // Convert to grayscale if not already
    Mat gray;
    if (image.channels() == 3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    // Apply slight Gaussian blur to reduce noise
    Mat smoothed;
    GaussianBlur(gray, smoothed, Size(3, 3), 0);

    // Define threshold value
    int threshValue = BACKTRACKING_THRESHOLD;

    // Create a matrix for segmentation
    Mat binary;
    threshold(smoothed, binary, threshValue, 255, THRESH_BINARY);
    
    // Create a copy for the result
    Mat segmented = binary.clone();
    
    // Choose a starting point for segmentation (center of image)
    int startX = gray.cols / 2;
    int startY = gray.rows / 2;
    int oldColor = binary.at<uchar>(startY, startX);
    int newColor = 128; // Mid-gray for marking the segmented region
    
    // Use a non-recursive approach to avoid stack overflow
    Mat visited = Mat::zeros(binary.size(), CV_8UC1);
    queue<Point> q;
    q.push(Point(startX, startY));
    visited.at<uchar>(startY, startX) = 1;
    
    // Define 8-directional connectivity (including diagonals)
    int dx[] = {1, -1, 0, 0, 1, -1, 1, -1};
    int dy[] = {0, 0, 1, -1, 1, -1, -1, 1};
    
    while (!q.empty()) {
        Point p = q.front();
        q.pop();
        
        // If the pixel matches the old color, mark it with the new color
        if (binary.at<uchar>(p.y, p.x) == oldColor) {
            segmented.at<uchar>(p.y, p.x) = newColor;
            
            // Check all 8 neighbors
            for (int i = 0; i < 8; i++) {
                int nx = p.x + dx[i];
                int ny = p.y + dy[i];
                
                // Boundary check
                if (nx >= 0 && ny >= 0 && nx < binary.cols && ny < binary.rows && 
                    !visited.at<uchar>(ny, nx)) {
                    
                    // For diagonal neighbors, check if both adjacent pixels are valid
                    bool isValid = true;
                    if (i >= 4) { // Diagonal directions
                        // Check horizontal neighbor
                        if (!visited.at<uchar>(p.y, nx)) {
                            isValid = binary.at<uchar>(p.y, nx) == oldColor;
                        }
                        // Check vertical neighbor
                        if (isValid && !visited.at<uchar>(ny, p.x)) {
                            isValid = binary.at<uchar>(ny, p.x) == oldColor;
                        }
                    }
                    
                    if (isValid) {
                        visited.at<uchar>(ny, nx) = 1;
                        q.push(Point(nx, ny));
                    }
                }
            }
        }
    }
    
    // Apply light morphological operations to clean up the result
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(segmented, segmented, MORPH_CLOSE, kernel);
    
    // Apply color map for better visualization
    Mat colored;
    applyColorMap(segmented, colored, COLORMAP_JET);
    
    return colored;
}

// Add update function for edge enhanced backtracking
static void update_backtracking_edge_enhanced_segmentation() {
    if (filename == NULL || input_image.empty()) {
        return;
    }

    try {
        auto start_time = chrono::high_resolution_clock::now();
        
        Mat processed_image = backtrackingEdgeEnhancementSegmentation(input_image);
        
        auto end_time = chrono::high_resolution_clock::now();
        double elapsed_time = chrono::duration<double, milli>(end_time - start_time).count();

        if (!processed_image.empty()) {
            string temp_filename = string(filename) + "_processed.jpg";
            imwrite(temp_filename, processed_image);

            GdkPixbuf *pixbuf = gdk_pixbuf_new_from_file_at_scale(temp_filename.c_str(), 
                                                             400, 400, 
                                                             TRUE, 
                                                             NULL);
            if (pixbuf) {
                gtk_image_set_from_pixbuf(GTK_IMAGE(processed_image_view), pixbuf);
                g_object_unref(pixbuf);
                
                gtk_label_set_text(GTK_LABEL(status_label), 
                    g_strdup_printf("Processing Time: %.2f ms", elapsed_time));
                
                gtk_label_set_text(GTK_LABEL(threshold_label), 
                    g_strdup_printf("Parameters:\nBacktracking threshold: %d\nEdge enhancement: Canny + Adaptive", 
                    BACKTRACKING_THRESHOLD));
            }
        }
    } catch (const cv::Exception& e) {
        gtk_label_set_text(GTK_LABEL(status_label), g_strdup_printf("Error: %s", e.what()));
    }
}



