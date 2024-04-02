using System;
using System.Windows.Input;

namespace MURDOC.ViewModel
{
    public class RelayCommand : ICommand
    {
        private readonly Action _execute;

        public event EventHandler CanExecuteChanged;

        public RelayCommand(Action execute)
        {
            _execute = execute ?? throw new ArgumentNullException(nameof(execute));
        }

        public bool CanExecute(object parameter)
        {
            return true; // For simplicity, always allow execution
        }

        public void Execute(object parameter)
        {
            _execute();
        }
    }
}