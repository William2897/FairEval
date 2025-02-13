export interface UserRole {
  role: 'ADMIN' | 'ACADEMIC';
  discipline: string;
}

export interface User {
  id: string;
  username: string;
  email: string;
  first_name: string;
  last_name: string;
  role?: UserRole;
}