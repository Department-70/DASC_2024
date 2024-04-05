﻿using System;
using System.ComponentModel;
using System.Windows.Input;
using Microsoft.Win32;
using System.Windows.Media.Imaging;
using System.IO;

namespace MURDOC.ViewModel
{
    public class MainWindowViewModel : INotifyPropertyChanged
    {
        #region Private Variables

        private string _selectedImagePath;

        private string _selectedImageFileName;

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
            }
        }

        /// <summary>
        /// Getter/Setter for the user selected image file name.
        /// </summary>
        public string SelectedImageFileName
        {
            get { return _selectedImageFileName; }
            private set
            {
                _selectedImageFileName = value;
                OnPropertyChanged(nameof(SelectedImageFileName));
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
            }
        }

        private string _rn50MPIcircle = "Resources/empty_circle.png";
        public string RN50MPIcircle
        {
            get { return _rn50MPIcircle; }
            set
            {
                _rn50MPIcircle = value;
                OnPropertyChanged(nameof(RN50MPIcircle));
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

        /// <summary>
        /// Constructor
        /// </summary>
        public MainWindowViewModel()
        {
            _exitCommand = new RelayCommand(ExecuteExitCommand);

            _browseCommand = new RelayCommand(ExecuteBrowseCommand);
            _selectedImageCommand = new RelayCommand(LoadImage);
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
            // TODO: Add logic for new command
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
            // TODO: Add logic for save command
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
        /// 
        /// </summary>
        private void ExecuteRunCommand()
        {

        }

        /// <summary>
        /// Loads an image from the user selected image path.
        /// </summary>
        private void LoadImage()
        {
            if (!string.IsNullOrEmpty(SelectedImagePath))
            {
                SelectedImage = new BitmapImage(new Uri(SelectedImagePath));
            }
        }
        
        /// <summary>
        /// Updates the GUI to display the user selected image file name
        /// </summary>
        private void UpdateSelectedImageFileName()
        {
            SelectedImageFileName = Path.GetFileName(SelectedImagePath);
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
