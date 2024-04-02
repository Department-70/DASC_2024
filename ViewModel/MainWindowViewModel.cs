using System;
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

        /// <summary>
        /// 
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
        /// 
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
        /// 
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

        /// <summary>
        /// 
        /// </summary>
        public MainWindowViewModel()
        {
            _exitCommand = new RelayCommand(ExecuteExitCommand);

            _browseCommand = new RelayCommand(ExecuteBrowseCommand);
            _selectedImageCommand = new RelayCommand(LoadImage);
        }

        /// <summary>
        /// 
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
        /// 
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
        private void LoadImage()
        {
            if (!string.IsNullOrEmpty(SelectedImagePath))
            {
                SelectedImage = new BitmapImage(new Uri(SelectedImagePath));
            }
        }
        
        /// <summary>
        /// 
        /// </summary>
        private void UpdateSelectedImageFileName()
        {
            SelectedImageFileName = Path.GetFileName(SelectedImagePath);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="propertyName"></param>
        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
