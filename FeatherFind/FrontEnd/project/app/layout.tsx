import './globals.css';
import type { Metadata } from 'next';
import { Exo_2 } from 'next/font/google';
import { ThemeProvider } from '@/components/theme-provider';

const exo2 = Exo_2({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'FeatherFind - Bird Identification Assistant',
  description: 'Your AI-powered bird identification and information assistant',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${exo2.className} antialiased`}>
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem={false}>
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}