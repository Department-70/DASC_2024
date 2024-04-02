using System;
using System.ComponentModel;
using System.Windows.Input;
using Microsoft.Win32;
using System.Windows.Media.Imaging;

namespace MURDOC.ViewModel
{
    public class MainWindowViewModel : INotifyPropertyChanged
    {
        #region private variables

        private string _selectedImagePath;

        private BitmapImage _selectedImage;
                
        private readonly ICommand _exitCommand;

        private readonly ICommand _newCommand;

        private readonly ICommand _saveCommand;

        private readonly ICommand _openCommand;

        private readonly ICommand _browseCommand;

        private readonly ICommand _selectedImageCommand;

        #endregion

        public event PropertyChangedEventHandler PropertyChanged;

        #region ICommands

        public ICommand ExitCommand => _exitCommand;

        public ICommand NewCommand => _newCommand;

        public ICommand SaveCommand => _saveCommand;

        public ICommand OpenCommand => _openCommand;

        public ICommand BrowseCommand => _browseCommand;

        public ICommand SelectedImageCommand => _selectedImageCommand;

        #endregion

        public string SelectedImagePath
        {
            get { return _selectedImagePath; }
            set
            {
                _selectedImagePath = value;
                OnPropertyChanged(nameof(SelectedImagePath));
            }
        }

        public BitmapImage SelectedImage
        {
            get { return _selectedImage; }
            set
            {
                _selectedImage = value;
                OnPropertyChanged(nameof(SelectedImage));
            }
        }

        public MainWindowViewModel()
        {
            _exitCommand = new RelayCommand(ExecuteExitCommand);

            _browseCommand = new RelayCommand(ExecuteBrowseCommand);
            _selectedImageCommand = new RelayCommand(LoadImage);
        }

        private void ExecuteExitCommand()
        {
            Console.WriteLine("In ExecuteExitCommand()");

            // Add logic to exit the application
            Environment.Exit(0);
        }

        private void ExecuteNewCommand() 
        { 
            // TODO: Add logic for new command
        }

        private void ExecuteOpenCommand() 
        { 
            // TODO: Add logic for open command
        }

        private void ExecuteSaveCommand() 
        { 
            // TODO: Add logic for save command
        }

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

        private void LoadImage()
        {
            if (!string.IsNullOrEmpty(SelectedImagePath))
            {
                SelectedImage = new BitmapImage(new Uri(SelectedImagePath));
            }
        }

        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
