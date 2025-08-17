import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    mode: 'dark',
    background: {
      default: '#1C1B2A',    // slightly lighter than navbar
      paper: '#2D2A5A',      // same as your navbar background
    },
    text: {
      primary: '#F3F3FF',    // same as navbar text
      secondary: '#9290C3',  // hover color
    },
    primary: {
      main: '#9290C3',
    },
  },
  typography: {
    fontFamily: 'Arial, Helvetica, sans-serif',
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: '#050a1f',
          color: '#F3F3FF',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundColor: '#2D2A5A',
          color: '#F3F3FF',
        },
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          backgroundColor: '#050a1f',
          color: '#F3F3FF',
          borderColor: '#F3F3FF',
        },
        notchedOutline: {
          borderColor: '#9290C3',
        },
        input: {
          color: '#F3F3FF',
        },
      },
    },
    MuiInputLabel: {
      styleOverrides: {
        root: {
          color: '#F3F3FF',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          color: '#F3F3FF',
          border: '1px solid #9290C3',
          backgroundColor: '#2D2A5A',
          '&:hover': {
            backgroundColor: '#3E3C70',
          },
        },
      },
    },
    MuiMenu: {
      styleOverrides: {
        paper: {
          backgroundColor: '#2D2A5A',
          color: '#F3F3FF',
        },
      },
    },
    MuiListItemText: {
      styleOverrides: {
        primary: {
          color: '#F3F3FF',
        },
      },
    },
    MuiCheckbox: {
      styleOverrides: {
        root: {
          color: '#9290C3',
        },
      },
    },
  },
});

export default theme;
