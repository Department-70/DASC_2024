using Microsoft.Win32;
using Python.Runtime; // Ensure this namespace is recognized without errors
using System;
using System.ComponentModel;
using System.IO;
using System.Windows.Input;
using System.Windows.Media.Imaging;

namespace MURDOC.ViewModel
{
    public class MainWindowViewModel : INotifyPropertyChanged
    {
        #region Private Variables

        private string _selectedImageFileName;  // filename with file type

        private string _selectedImageName;      // filename without file type

        private BitmapImage _selectedImage;

        private readonly ICommand _exitCommand;

        private readonly ICommand _newCommand;

        private readonly ICommand _saveCommand;

        private readonly ICommand _openCommand;

        private readonly ICommand _browseCommand;

        private readonly ICommand _selectedImageCommand;

        private readonly ICommand _runCommand;

        #endregion

        public event PropertyChangedEventHandler PropertyChanged;

        #region ICommands

        public ICommand ExitCommand => _exitCommand;

        public ICommand NewCommand => _newCommand;

        public ICommand SaveCommand => _saveCommand;

        public ICommand OpenCommand => _openCommand;

        public ICommand BrowseCommand => _browseCommand;

        public ICommand SelectedImageCommand => _selectedImageCommand;

        public ICommand RunCommand => _runCommand;

        #endregion

        private string _selectedImagePath;

        /// <summary>
        /// Getter/Setter for the user selected image path.
        /// </summary>
        public string SelectedImagePath
        {
            get { return _selectedImagePath; }
            set
            {
                _selectedImagePath = value;
                OnPropertyChanged(nameof(SelectedImagePath));
                UpdateSelectedImageFileName(); // Update SelectedImageFileName when SelectedImagePath changes

                // Enable or disable the Run button based on whether an image is selected
                IsRunButtonEnabled = !string.IsNullOrEmpty(value);
            }
        }

        /// <summary>
        /// Getter/Setter for the user selected image file name.
        /// </summary>
        public string SelectedImageFileName
        {
            get => _selectedImageFileName; 
            private set
            {
                _selectedImageFileName = value;
                OnPropertyChanged(nameof(SelectedImageFileName));
            }
        }

        /// <summary>
        /// Getter/Setter for the user selected image file name without the file extension.
        /// </summary>
        public string SelectedImageName
        {
            get => _selectedImageName;
            private set
            {
                _selectedImageName = value;
                OnPropertyChanged(nameof(SelectedImageName));
            }
        }

        /// <summary>
        /// Returns the user selected image to be displayed on the GUI.
        /// </summary>
        public BitmapImage SelectedImage
        {
            get { return _selectedImage; }
            set
            {
                _selectedImage = value;
                OnPropertyChanged(nameof(SelectedImage));

                // TODO: Reset all Model Traversal Progress circles to empty

                // TODO: Clear all of the Model Traversal Results - except for Input Image
            }
        }

        public BitmapImage NoSelectedImage
        {
            get { return new BitmapImage(new Uri("pack://application:,,,/MURDOC;component/Assets/image_placeholder.png")); ; }
            set { }
        }

        #region ResNet50 Off-Ramp Images
        /// <summary>
        /// Setter will set the appropriate off-ramp images dependant on:
        ///         (1) if the model has run
        ///         (2) if the off-ramp image exists in the appropriate folder
        /// </summary>
        private string _resNet50ConvImagePath;
        public string ResNet50ConvImagePath
        {
            get { return _resNet50ConvImagePath; }
            set
            {
                if (_resNet50ConvImagePath != value)
                {
                    _resNet50ConvImagePath = value;
                    OnPropertyChanged(nameof(ResNet50ConvImagePath));

                    // Load and set the image directly to ResNet50Conv
                    LoadResNet50ConvImage();
                }
            }
        }

        private BitmapImage _resNet50ConvImage;
        public BitmapImage ResNet50Conv
        {
            get { return _resNet50ConvImage; }
            set
            {
                _resNet50ConvImage = value;
                OnPropertyChanged(nameof(ResNet50Conv));
            }
        }
        #endregion

        /// <summary>
        /// Disables/Enables the Run button 
        /// Relies upon if a user selects an image or not
        /// </summary>
        private bool _isRunButtonEnabled;
        public bool IsRunButtonEnabled
        {
            get { return _isRunButtonEnabled; }
            set
            {
                _isRunButtonEnabled = value;
                OnPropertyChanged(nameof(IsRunButtonEnabled));
            }
        }

        #region Model Traversal Progress circles
        private string _rn50MPIcircle = "Assets/empty_circle.png";
        public string RN50MPIcircle
        {
            get { return _rn50MPIcircle; }
            set
            {
                _rn50MPIcircle = value;
                OnPropertyChanged(nameof(RN50MPIcircle));
            }
        }

        private string _rn50ResultsCircle = "Assets/empty_circle.png";
        public string RN50ResultsCircle
        {
            get { return _rn50ResultsCircle; }
            set
            {
                _rn50ResultsCircle = value;
                OnPropertyChanged(nameof(RN50ResultsCircle));
            }
        }

        private string _rNetMPIcircle = "Assets/empty_circle.png";
        public string RNetMPIcircle
        {
            get { return _rNetMPIcircle; }
            set
            {
                _rNetMPIcircle = value;
                OnPropertyChanged(nameof(RNetMPIcircle));
            }
        }

        private string _rNetResultsCircle = "Assets/empty_circle.png";
        public string RNetResultsCircle
        {
            get { return _rNetResultsCircle; }
            set
            {
                _rNetResultsCircle = value;
                OnPropertyChanged(nameof(RNetResultsCircle));
            }
        }

        private string _eDD7MPICircle = "Assets/empty_circle.png";
        public string EDD7MPIcircle
        {
            get { return _eDD7MPICircle; }
            set
            {
                _eDD7MPICircle = value;
                OnPropertyChanged(nameof(EDD7MPIcircle));
            }
        }

        private string _eDD7ResultsCircle = "Assets/empty_circle.png";
        public string EDD7ResultsCircle
        {
            get { return _eDD7ResultsCircle; }
            set
            {
                _eDD7ResultsCircle = value;
                OnPropertyChanged(nameof(EDD7ResultsCircle));
            }
        }

        private string _finalResultCircle = "Assets/empty_circle.png";
        public string FinalResultsCircle
        {
            get { return _finalResultCircle; }
            set
            {
                _finalResultCircle = value;
                OnPropertyChanged(nameof(FinalResultsCircle));
            }
        }

        private string _rn50ModelStatus;

        public string RN50ModelStatus
        {
            get { return _rn50ModelStatus; }
            set
            {
                _rn50ModelStatus = value;
                UpdateStepCompletionStatus(); // Update step completion status when model status changes
                OnPropertyChanged(nameof(RN50ModelStatus));
            }
        }
        #endregion

        /// <summary>
        /// Constructor
        /// </summary>
        public MainWindowViewModel()
        {
            IsRunButtonEnabled = false; // Disable Run button initially

            LoadImage();

            LoadResNet50ConvImage();

            _exitCommand = new RelayCommand(ExecuteExitCommand);

            _browseCommand = new RelayCommand(ExecuteBrowseCommand);
            _selectedImageCommand = new RelayCommand(LoadImage);

            _runCommand = new RelayCommand(ExecuteRunCommand);
        }

        /// <summary>
        /// Closes the application.
        /// </summary>
        private void ExecuteExitCommand()
        {
            Console.WriteLine("In ExecuteExitCommand()");

            // Add logic to exit the application
            Environment.Exit(0);
        }

        /// <summary>
        /// 
        /// </summary>
        private void ExecuteNewCommand()
        {
            // TODO: Add logic for new command - reset everything on the screen
        }

        /// <summary>
        /// 
        /// </summary>
        private void ExecuteOpenCommand()
        {
            // TODO: Add logic for open command
        }

        /// <summary>
        /// 
        /// </summary>
        private void ExecuteSaveCommand()
        {
            // TODO: Add logic for save command - Save the models 'visualization' as a PDF
        }

        /// <summary>
        /// Executes the BrowseCommand to open a file dialog for selecting an image file.
        /// </summary>
        private void ExecuteBrowseCommand()
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Image Files|*.jpg;*.jpeg;*.png;*.gif;*.bmp";
            if (openFileDialog.ShowDialog() == true)
            {
                SelectedImagePath = openFileDialog.FileName;
                LoadImage();
            }
        }

        /// <summary>
        /// Run the FACE model (submodels include ResNet50, RankNet, and EfficientDet-D7)
        /// </summary>
        private void ExecuteRunCommand()
        {
            // Need to handle the scenario by preventing the run when SelectedImagePath has no image selected

            // Initialize Python engine
            if (!PythonEngine.IsInitialized)
            {
                InitializePythonEngine();
            }

            using (Py.GIL())
            {
                dynamic sys = Py.Import("sys");
                dynamic os = Py.Import("os");

                // Add the directory containing your Python script to Python's sys.path
                string scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"..\..\..\", "Model", "XAI_ResNet50.py");
                sys.path.append(os.path.dirname(scriptPath));

                // ResNet50
                try
                {
                    // Import your Python script module
                    dynamic script = Py.Import("XAI_ResNet50");

                    // Update the RN50MPIcircle to green circle to show model is running
                    RN50MPIcircle = "Assets/filled_circle.png";

                    // Call the process_image_with_resnet50 function from your Python script
                    script.process_image_with_resnet50(SelectedImagePath);

                    // TODO: Load the Off-ramp images
                    string executableDir = AppDomain.CurrentDomain.BaseDirectory;
                    string folderPath = Path.Combine(executableDir, "resnet50_output", _selectedImageName);

                    // Update the ResNet50ConvImagePath to trigger UI update
                    string imagePath = Path.Combine(folderPath, _selectedImageName + "_initial_conv_feature_map.png");
                    ResNet50ConvImagePath = imagePath;
                    OnPropertyChanged(nameof(ResNet50Conv)); // Trigger UI update for ResNet50Conv
                }
                catch (PythonException exception)
                {
                    // Probably should indicate somewhere on the GUI that something went wrong
                    Console.WriteLine("Exception occured: " + exception);
                }

                // Update the RN50ResultsCircle to green circle to show the model is done
                RN50ResultsCircle = "Assets/filled_circle.png";

                // TODO: Populate the ResNetConv, ResNet50Block1-4, and ResNet50Output images

                // TODO: Run the RankNet model

                // TODO: Run the EfficientDetD7 model

                // TODO: Display the final prediction
            }
        }

        /// <summary>
        /// Initializes Python Engine with the default Visual Studio Python DLL location
        /// </summary>
        private void InitializePythonEngine()
        {
            try
            {
                string pythonDll = @"C:\Users\pharm\AppData\Local\Programs\Python\Python39\python39.dll";
                Environment.SetEnvironmentVariable("PYTHONNET_PYDLL", pythonDll);

                // Initialize will fail if configuration manager is not set up or ran with x64 since the above python Dll is 64 bit
                PythonEngine.Initialize();
            }
            catch (TypeInitializationException tiex)
            {
                // Log or display the type initialization exception message and inner exception
                Console.WriteLine("Type Initialization Exception: " + tiex.Message);
                Console.WriteLine("Inner Exception: " + tiex.InnerException?.Message);
            }
            catch (Exception ex)
            {
                // Log or display any other exceptions during initialization
                Console.WriteLine("Error initializing Python engine: " + ex.Message);
            }
        }

        /// <summary>
        /// Loads an image from the user selected image path or sets a default placeholder image.
        /// </summary>
        private void LoadImage()
        {
            if (!string.IsNullOrEmpty(SelectedImagePath))
            {
                SelectedImage = new BitmapImage(new Uri(SelectedImagePath));
            }
            else
            {
                // Set the default placeholder image
                SelectedImage = new BitmapImage(new Uri("pack://application:,,,/MURDOC;component/Assets/image_placeholder.png"));
            }
        }

        /// <summary>
        /// Loads an image from the user selected image path or sets a default placeholder image.
        /// </summary>
        private void LoadResNet50ConvImage()
        {
            if (!string.IsNullOrEmpty(ResNet50ConvImagePath))
            {
                ResNet50Conv = new BitmapImage(new Uri(ResNet50ConvImagePath));
            }
            else
            {
                // Set the default placeholder image
                ResNet50Conv = new BitmapImage(new Uri("pack://application:,,,/MURDOC;component/Assets/image_placeholder.png"));
            }
        }

        /// <summary>
        /// Updates the GUI to display the user selected image file name
        /// </summary>
        private void UpdateSelectedImageFileName()
        {
            if (SelectedImagePath != "Assets/image_placeholder.png")
            {
                SelectedImageFileName = Path.GetFileName(SelectedImagePath);
                SelectedImageName = Path.GetFileNameWithoutExtension(SelectedImagePath);
            }
        }

        // In a method where you determine the completion status of each step, set the appropriate image source:
        private void UpdateStepCompletionStatus()
        {
            // Example logic (replace with your actual logic):
            bool rn50Completed = IsRN50ModelCompleted();
            RN50MPIcircle = rn50Completed ? "Resources/filled_circle.png" : "Resources/empty_circle.png";

            // Update other step completion properties similarly...
        }

        // Assuming you have a method to determine the completion status of each step:
        private bool IsRN50ModelCompleted()
        {
            // Example logic to determine if RN50 model is completed
            return _rn50ModelStatus == "Completed";
        }

        /// <summary>
        /// Invokes the PropertyChanged event to notify subscribers of a property change.
        /// </summary>
        /// <param name="propertyName">The name of the property that changed.</param>
        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
