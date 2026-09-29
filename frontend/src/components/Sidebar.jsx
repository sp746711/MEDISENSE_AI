import { NavLink } from 'react-router-dom';

export default function Sidebar() {
  return (
    <aside className="sidebar" aria-label="Quick actions">
      <NavLink to="/assessment/new">+ New Assessment</NavLink>
      <NavLink to="/history">History</NavLink>
      <NavLink to="/assistant">AI Assistant</NavLink>
      <NavLink to="/doctors">Doctors</NavLink>
      <NavLink to="/facilities">Facilities</NavLink>
      <NavLink to="/appointments">Appointments</NavLink>
    </aside>
  );
}
